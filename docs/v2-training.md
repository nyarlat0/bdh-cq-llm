# Architecture v2 and the 1.05B-token training run

This is the operational specification for the second model. It exists beside
the original run: v1 checkpoints, tokenizer and packed shards are not modified
or imported because the vocabulary and tensor shapes are incompatible.

The production config is [`configs/rx6700-v2.json`](../configs/rx6700-v2.json).
The fixed-budget architecture and RoPE pilots are complete; their measurements,
plots, limitations and final selection are recorded in
[`v2-pilot-results.md`](v2-pilot-results.md). This remains an experimental local
architecture, not a claim about published BDH-CQ results.

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
| state gates, multi-head Attention Residual and CQ decay | <0.03M |
| total | 22,043,648 |

The total model is smaller, but the recurrent body is four times wider than
v1. This is a much better parameter allocation for a small base LM.

## 2. Exact architecture

The production candidate uses `D=512`, recurrent depth `L=8`, `H=8` heads and
`Q=768` positive features per head (`H*Q=6144`). All eight depths reuse the
same block weights.

### 2.1 RoPE

Production rotates the first 384 coordinates in every 768-wide head. The
isolated 64/192/384 sweep selected this width: it reached 7.4733 stateful loss
versus 7.4856 at width 64, with a 1.8% throughput cost and no parameter-count
change. Local causal attention still covers the complete 256-token chunk; RoPE
width changes positional capacity, not context length. See section 6 and the
complete [pilot report](v2-pilot-results.md).

### 2.2 Full neuron state across recurrent depth

Let `Z_l` be the ordinary lifted/gated BDH activation with shape
`[B,H,N,Q]`. V2 carries a full state `S_l` of exactly that shape, but exports
only its current change through the narrow communication channel:

```text
G_l = ReLU(W_qk X_l + alpha ⊙ RMSNorm(S_(l-1)))

if S_(l-1) is absent:
    DeltaS_l = Z_l
    S_l = Z_l
else:
    u_l = sigmoid(W_u X_l + raw_u)          [B,H,N,Q]
    DeltaS_l = u_l ⊙ (Z_l - S_(l-1))       [B,H,N,Q]
    S_l = S_(l-1) + DeltaS_l                [B,H,N,Q]

Y_l = LayerNorm(W_out concat(DeltaS_l))      [B,N,D]
```

`raw_u` is `[H,Q]`, while `W_u X` remains the cheap `[B,N,H]` projection and
broadcasts over `Q`. Different coordinates in one head can therefore learn
different baseline timescales for only `H*Q` extra parameters. `raw_u` starts
at `logit(0.2)`. The first state is `Z_0`, not `0.2 Z_0`; the EMA/update begins
only when an actual previous state exists.

Direct injection is also per-neuron:
`alpha = tanh(raw_alpha)`, `raw_alpha: [H,Q]`. V2 configs initialize
`alpha=0.05`, a small live path near the old no-injection behavior. No
`[H*Q,H*Q]` matrix exists. Most importantly, `proj_out` consumes `DeltaS`, not
`S_l`: accumulated wide information is not projected and residually re-added
at every depth.

For ordinary token chunks this state is local to one `Bdh::forward` call. It
survives eight recurrent depths, then is discarded. Carrying `[B,H,N,Q]`
between chunks would make state size depend on chunk length, retain
token-aligned activations indefinitely and duplicate CQ. During
`Stage::Think(R)`, however, `N=1`: an explicit `LatentWorkspace [B,H,1,Q]`
survives all `R` outer reasoning iterations with gradients intact. It resets
at the next independent Think chain and never enters `Memory.fast_weights`.

### 2.3 Multi-Head Attention Residual (MHAR)

When enabled, v2 replaces `X_l + Y_l` with learned attention over the seed
state and all block deltas produced so far:

```text
history_l = [X_0, Y_0, ..., Y_l]                    [B,N,K,D]
keys      = reshape(RMSNorm_D(history_l), K, R, D/R)
a_(l,r)   = softmax_K(sum_d keys_(K,r,d) p_(r,d))
X_(l+1)   = LayerNorm(concat_r sum_K a_(l,r,K) history_(K,r))
```

Every `Y_l` in history is `W_out(DeltaS_l)`, never a projection of cumulative
`S_l`. The production candidate uses `R=8` routing heads, so each independently
chooses a depth mixture for a contiguous 64-feature subspace of `D=512`.
RMSNorm is applied over the complete `D` row before splitting it; the softmax
is over history depth separately for every routing head. There is no
`1/sqrt(d)` factor. The query is tied across recurrent BDH depth as requested,
but contains all eight subqueries. It is zero-initialized, which makes every
head a uniform history average at step zero. `R=1` is exactly the old
single-query Attention Residual and has the same parameter count as `R=8`.

This follows the Multi-Head Attention Residuals construction rather than
simply copying the older single-query router. It is not a disclosed part of
Pathway's proprietary BDH-CQ update. The 2×2 architecture grid still tests
whether any attention residual helps, and an additional H=1 control isolates
whether multi-head routing itself is useful on this model and corpus.

### 2.4 Decaying CQ memory

Each recurrent depth still owns one fixed fast-weight matrix
`M_l: [B,H,Q,D]`. Its update is now:

```text
rho_(h,q) = sigmoid(raw_rho_(h,q))
M_l <- rho ⊙ M_l + K_l^T V_l
```

`raw_rho` is an unconstrained `[H,Q]` tensor, and every sigmoid starts at
0.995. If a coordinate remained at that value, its half-life would be about
138 chunks or 35.4K source tokens.
Decay prevents the unbounded magnitude growth of a purely additive sum while
allowing individual neuron coordinates to learn longer or shorter retention.
All rank-1 parameters plus the semantic `[H,Q]` arrays `raw_rho`, `raw_u` and
`raw_alpha` are excluded from AdamW decay; projection matrices and embedding
tables retain the configured decay. Logs report CQ RMS/maximum plus
min/mean/max `rho`, base `u`, and bounded `alpha` rather than thousands of
values.

### 2.5 Narrow recurrent normalization

V2 applies parameter-free LayerNorm after every shared-block application:

```text
X_(l+1) = LayerNorm(X_l + Y_l)              additive path
X_(l+1) = LayerNorm(MHAR([X_0,Y_0,...]))    routing path
```

The legacy/public-reconstruction default remains `normalize_each_depth=false`
and applies only the final LayerNorm. No affine parameters are introduced.

### 2.6 Tied vocabulary matrix

The output logits are `hidden @ embedding^T / sqrt(D)`; the scale keeps random
initial logits variance-bounded after hidden LayerNorm. There is no second
LM-head matrix. The tokenizer has 24,576 entries, keeping the tied matrix at
12.58M parameters while preserving byte fallback and lossless Russian UTF-8.

## 3. What “context” means

There are three distinct horizons:

| mechanism | production value | crosses chunks? |
|---|---:|---|
| exact local causal attention | 256 tokens | no |
| gradient horizon through CQ (truncated BPTT) | 1 chunk = 256 tokens | gradients only |
| CQ value lifetime | until `<|doc|>` or work-block reset | yes, detached every chunk |

Four stable lanes are trained concurrently in one physical `[4,256]` forward.
Every lane receives consecutive chunks from its own stripe. `Memory` stores a
batch of four independent CQ rows and four independent RoPE offsets. A
256-sequence work block gives each lane up to 64 adjacent chunks, or 16,384
tokens, before the mandatory shuffle-boundary reset.

If `<|doc|>` appears at different positions in different lanes, row-major
document-start flags produce per-token positions and three masks: local
attention cannot cross the boundary, tokens at/after it cannot read the old CQ
row, and only the final document in the chunk is written back. Other lanes are
untouched. This preserves exact lane semantics without four serial B=1 calls;
tests compare the batched result with independent reference forwards.

CQ therefore supplements local context but does not turn the model into exact
4096-token softmax attention. Tokens older than 256 are available only through
the learned compressed associations in `M_l`.

The detach interval is intentionally one chunk on the 12 GiB RX 6700 XT. An
earlier eight-chunk pilot retained a 2,048-token autograd graph and requested
about 26.4 GiB of Vulkan buffers. amdgpu consequently placed roughly 14.8 GiB
in GTT/system memory, cutting stateful throughput by about five times. CQ
*values* still cross every chunk and survive for the complete document; only
the gradient path into the preceding chunk is cut. This is the original
streaming schedule: `chunk -> Memory.detach() -> next chunk`.

Burn/CubeCL's default sliced allocator also keeps completely free pages at its
peak high-water mark unless an explicit cleanup is requested. The trainer now
calls backend cleanup after every optimizer update and after validation. It
also batches the sixteen scalar training losses on the GPU and performs one
CPU readback per update rather than one per microbatch. Linux/amdgpu logs add
`gpu_requested_mib`, `gpu_vram_mib` and `gpu_gtt_mib` after cleanup plus
`gpu_peak_*_mib` immediately before it. Peak GTT at or above 1 GiB prints an
explicit spill warning.

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
./scripts/prepare_v2_data.sh
```

With no argument the script creates an ignored, persistent
`.venv-tokenizer/`, installs only the Python dataset-reader dependencies, then
creates `artifacts/tokenizer-v2-24576.json` and
`datasets/packed/rx6700-v2-24576/`. It does not use `--force`; existing data is
never silently overwritten. Complete shards are validated and reused, so an
interrupted packing run can be resumed by executing the same command. The
generated manifest records `ficbook_metadata_included: false`.

## 6. Hardware width check and completed pilot experiments

First compare `H*Q` 4096, 5120 and 6144 on one complete stateful work block:

```console
python3 scripts/benchmark_v2_widths.py 0
```

The temporary runs live under `/tmp`. Select 6144 only if it fits with useful
VRAM margin and its sustained speed is acceptable; otherwise change
`dim_qk_heads` consistently in production and all pilot configs.

Then run the architecture grid and routing control:

```console
scripts/run_v2_pilots.sh 0
```

Each run processes 1,220 updates = 19,988,480 tokens. Pilot-only CQ activation
is accelerated to 5M so roughly 15M tokens exercise persistent memory. The
four architecture cases are additive, MHAR only, neuron state only, and both.
A fifth `attnres-state-h1` run is identical to the combined H=8 candidate except
for classic single-query routing. They have isolated run directories and the
launcher refuses to resume an old pilot, because unequal token budgets
invalidate the comparison.

Delta-state pilots use fresh `runs/rx6700-v2-delta-pilot-*` directories. The
older pilot directories are deliberately preserved but must not be compared:
they used both the previous cumulative-state semantics and, for the oldest
runs, eight-chunk graphs that spilled into GTT.

The completed architecture table selected the combined `attnres-state` model:
7.4856 stateful loss versus 7.6488 for the additive control, corresponding to
about 15.1% lower perplexity. H=8 and H=1 routing were effectively tied; H=8 is
retained because it has the same parameter count, negligible extra cost and
preserves per-subspace depth routing. Exact tables and curves are in the
[pilot report](v2-pilot-results.md).

After selecting the combined architecture, compare positional width alone:

```console
scripts/run_v2_rope_sweep.sh 0
```

The three configs `rx6700-v2-rope-{64,192,384}.json` are identical except for
`model.rotary_dim` and isolated `run_dir`. Each uses the same 1,220-update
(19,988,480-token) budget and accelerated 5M-token CQ activation as the main
pilots. Do not pool these runs into the A–E architecture table: the first table
selects state/routing mechanisms, while this second experiment selects only
RoPE width. The completed sweep selected 384/768 after it beat width 64 at
every stateful validation checkpoint. The production config now carries that
decision.

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
