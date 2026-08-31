# Architecture v2 and the 1.05B-token training run

This is the operational specification for the second model. It exists beside
the original run: v1 checkpoints, tokenizer and packed shards are not modified
or imported because the vocabulary and tensor shapes are incompatible.

The production config is [`configs/rx6700-v2.json`](../configs/rx6700-v2.json).
It is the current **hypothesis**, not a claim that all new mechanisms help. The
four fixed-budget pilots below are the acceptance gate before the long run.

## 1. Why this is a new model

The first run had `D=384`, depth 6, `H*Q=2048`, a 32,768-token vocabulary and
two separate 12.6M-parameter vocabulary matrices. Of its 27.5M parameters,
about 25.2M were therefore spent on input/output word pieces and only about
2.4M on the recurrent BDH computation. This explains much of the slow
held-out-loss improvement: the model saw plenty of tokens, but the body was a
severe bottleneck. Raw per-step loss also mixed sources and document
difficulties, so its large short-term oscillations were expected.

V2 uses a smaller vocabulary and tied embeddings. Its approximate allocation
is:

| component | parameters |
|---|---:|
| one tied `24576 × 512` vocabulary matrix | 12.58M |
| shared `W_qk`, `W_up`, `W_out` at `H*Q=6144` | 9.44M |
| state gates, Attention Residual and CQ decay | <0.01M |
| total | about 22.03M |

The total model is smaller, but the recurrent body is four times wider than
v1. This is a much better parameter allocation for a small base LM.

## 2. Exact architecture

The production candidate uses `D=512`, recurrent depth `L=8`, `H=8` heads and
`Q=768` positive features per head (`H*Q=6144`). All eight depths reuse the
same block weights.

### 2.1 RoPE

Only the first 64 coordinates in every 768-wide head receive RoPE. The old
implicit rule rotated half of each head. An explicit narrow positional slice
leaves most of the positive workspace semantic and avoids turning hundreds of
ReLU coordinates into signed rotations. Local causal attention still covers
the complete 256-token chunk.

### 2.2 Full neuron state across recurrent depth

Let `Z_l` be the ordinary lifted/gated BDH activation with shape
`[B,H,N,Q]`. V2 optionally carries another tensor of exactly that shape:

```text
u_l = sigmoid(W_u X_l + logit(0.99))       [B,H,N,1]
S_l = (1 - u_l) S_(l-1) + u_l Z_l          [B,H,N,Q]
Y_l = LayerNorm(W_out concat(S_l))          [B,N,D]
```

Before constructing Q/K at the next depth, normalized `S_l` can also be added
to the positive gate through one learned strength per head. Those strengths
start at zero. `W_u` starts at zero and the fixed offset makes `u≈0.99`, so
the initial network is close to the stateless block instead of accidentally
halving its activation.

This state is deliberately local to one `Bdh::forward` call. It survives the
eight recurrent depths of a chunk and receives gradients through them, then is
discarded. Carrying `[B,H,N,Q]` between chunks would make state size depend on
chunk length, retain token-aligned activations indefinitely and duplicate the
job of CQ. Only the fixed-size CQ matrices cross chunk boundaries.

### 2.3 Attention Residual

When enabled, v2 replaces `X_l + Y_l` with learned attention over the seed
state and all block deltas produced so far:

```text
history_l = [X_0, Y_0, ..., Y_l]
a_l       = softmax(<RMSNorm(history_l), p>)
X_(l+1)   = sum_i a_(l,i) history_(l,i)
```

The pseudo-query `p` is tied across depth and the language-model config uses no
cycle-distance bias. This matches the optional Attention Residual mechanism in
the public reconstruction. It is not a disclosed part of Pathway's proprietary
BDH-CQ update, which is why additive versus Attention Residual remains a pilot
axis.

### 2.4 Decaying CQ memory

Each recurrent depth still owns one fixed fast-weight matrix
`M_l: [B,H,Q,D]`. Its update is now:

```text
r_h = sigmoid(rho_h)
M_l <- r_h M_l + K_l^T V_l
```

`r_h` is learned independently per head and starts at 0.995. If it remained at
that value, its half-life would be about 138 chunks or 35.4K source tokens.
Decay prevents the unbounded magnitude growth of a purely additive sum while
allowing training to learn longer or shorter retention. Logs report CQ RMS and
absolute maximum so exploding state is visible.

### 2.5 Tied vocabulary matrix

The output logits are `hidden @ embedding^T / sqrt(D)`; the scale keeps random
initial logits variance-bounded after hidden LayerNorm. There is no second
LM-head matrix. The tokenizer has 24,576 entries, keeping the tied matrix at
12.58M parameters while preserving byte fallback and lossless Russian UTF-8.

## 3. What “context” means

There are three distinct horizons:

| mechanism | production value | crosses chunks? |
|---|---:|---|
| exact local causal attention | 256 tokens | no |
| gradient horizon through CQ (truncated BPTT) | 8 chunks = 2,048 tokens | gradients only |
| CQ value lifetime | until `<|doc|>` or work-block reset | yes, detached every 8 chunks |

Four stable lanes are trained concurrently. Every lane receives consecutive
chunks from its own stripe and owns a separate `Memory`; no document can leak
into another lane. A 256-sequence work block gives each lane up to 64 adjacent
chunks, or 16,384 tokens, before the mandatory shuffle-boundary reset. A
`<|doc|>` resets that lane earlier. The loader executes stateful lanes
independently because their document resets and RoPE offsets differ; this is a
correctness choice, and the hardware benchmark measures its real cost.

CQ therefore supplements local context but does not turn the model into exact
4096-token softmax attention. Tokens older than 256 are available only through
the learned compressed associations in `M_l`.

## 4. Data and schedule

No rating, tag, profanity, NSFW or semantic-content filter is applied. This is
separate from metadata removal:

- FineWeb contributes only its `text` column;
- Ficbook contributes only non-empty `parts[].clean_text` bodies;
- titles, descriptions, tags, ratings and chapter titles are not text;
- ru-classic contributes its text-file contents;
- the packer adds only structural `<|doc|>` markers.

The base model is raw next-token completion. No `user`, `assistant`, system
prompt or chat-template labels are inserted. Reserved role IDs remain unused
until a later, separately constructed chat-SFT dataset.

The unique packed training corpus remains 1B tokens. V2 processes 1.05B because
phase two deliberately replays 50M general-domain tokens:

| phase | source | processed tokens |
|---|---|---:|
| general | FineWeb | 650M |
| general | Ficbook first pass | 50M |
| general | ru-classic | 50M |
| focus | Ficbook second pass | 250M |
| focus replay | FineWeb prefix | 40M |
| focus replay | ru-classic prefix | 10M |

After whole-sequence and optimizer-boundary trimming, the deterministic
schedule is 1,049,985,024 target tokens and 64,086 optimizer updates. One
update contains `4 lanes × 16 microbatches × 256 = 16,384` target tokens.

CQ is disabled for the first 100M processed tokens. Phase-one LR warms for 10M
tokens to `3e-4`, then cosines to `8e-5`. At the approximately 750M phase
boundary it warms for 5M from `8e-5` to `1.2e-4`, then cosines to `3e-5`. The
focus re-warm prevents the old late-stage tiny LR from making the 250M Ficbook
pass nearly inert.

## 5. Data preparation

Use the provided wrapper (its first optional argument is the Python executable):

```console
scripts/prepare_v2_data.sh /tmp/bdh-cq-tokenizer-venv/bin/python
```

It creates `artifacts/tokenizer-v2-24576.json` and
`datasets/packed/rx6700-v2-24576/`. It does not use `--force`; existing data is
never silently overwritten. The generated manifest records
`ficbook_metadata_included: false`.

## 6. Hardware width check and 2×2 pilots

First compare `H*Q` 4096, 5120 and 6144 on one complete stateful work block:

```console
python3 scripts/benchmark_v2_widths.py 0
```

The temporary runs live under `/tmp`. Select 6144 only if it fits with useful
VRAM margin and its sustained speed is acceptable; otherwise change
`dim_qk_heads` consistently in production and all pilot configs.

Then run the architecture grid:

```console
scripts/run_v2_pilots.sh 0
```

Each run processes 1,220 updates = 19,988,480 tokens. Pilot-only CQ activation
is accelerated to 5M so roughly 15M tokens exercise persistent memory. The
four axes are additive, Attention Residual only, neuron state only, and both.
They have isolated run directories and the launcher refuses to resume an old
pilot, because unequal token budgets invalidate the comparison.

The final report uses stateful held-out loss, per-source BPB, finite-state
checks and median throughput. A difference below roughly 1% is weak evidence;
prefer the faster/simpler variant in that case. Copy only the winning
`attn_residual` and `gated_neuron_state` settings into the production config
before its run directory exists.

## 7. Production and monitoring

After the width and pilot decisions:

```console
cargo run --release --bin train_llm -- --config configs/rx6700-v2.json
```

Validation events are appended to `runs/rx6700-v2/train.jsonl`. They contain
memoryless and stateful loss, perplexity and decoded UTF-8 bits/byte separately
for FineWeb, Ficbook and classics. `checkpoints/latest.json` names the resume
checkpoint; `checkpoints/best.json` protects the best validated checkpoint
from normal pruning. Train loss is intentionally noisy and must not be used as
the sole convergence signal.

For a graceful stop:

```console
touch runs/rx6700-v2/STOP
```

The trainer saves at the next safe block boundary and exits. A first Ctrl+C
requests the same behavior. Remove `STOP` before resuming. Do not use SIGKILL
when a resumable checkpoint is required.

No script in this change starts the production run automatically.
