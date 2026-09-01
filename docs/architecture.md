# BDH-CQ architecture guide

This guide explains the executable public reconstruction in this repository
and connects it to the two research papers. Keep the fidelity boundary in
mind: the BDH-CQ paper describes the surrounding recurrent architecture but
does not publish the exact update function used by the evaluated system. The
tensor equations below describe `lucidrains/bdh-cq`, which is what the Rust
code can faithfully port.

## 1. Three related objects

It is easy to accidentally treat three different things as one model:

1. **Original BDH** introduces a high-dimensional sparse/positive neuron space,
   low-dimensional communication, fixed-size recurrent state, and weights
   shared across recurrent depth.
2. **The BDH-CQ paper** places a recurrent update function inside a continual
   learning and latent-reasoning system. It defines the interfaces but says
   its exact update rule and dimensions are proprietary.
3. **The public reconstruction** chooses a concrete update: causal linear
   attention whose history is compressed into additive `K^T V` fast weights.
   It also adds partial RoPE, latent-step supervision, and optional Attention
   Residuals.

This crate implements item 3, uses item 1 to explain why the construction
makes architectural sense, and uses item 2 to name the contextual and latent
state roles. Architecture v2 is a fourth, explicitly experimental layer on
top: delta-form wide state, per-neuron timescales/injection/retention,
depth-local normalization and MHAR are project hypotheses, not published
Pathway BDH-CQ mechanisms.

## 2. Shape glossary

| Symbol | Meaning | Typical upstream default |
|---|---|---:|
| `B` | batch size | experiment dependent |
| `N` | positions in the current chunk | up to 128 for ARC ingestion |
| `D` | low-dimensional token/value channel | 512 core, 384 ARC script |
| `H` | number of feature heads | 4 |
| `Q` | positive Q/K features per head | `(H*Q)/H` |
| `H*Q` | total high-dimensional feature count | 32,768 core, 2,048 ARC |
| `L` | recurrent depth, using the same block parameters | 8 core, 4 ARC |
| `R` | continuous latent iterations | chosen at run time |
| `V` | vocabulary size | 14 for synthetic ARC |

The original BDH paper calls the large neuron dimension `n` and the smaller
communication dimension `d`. Here, `n` corresponds roughly to `H*Q`, and `d`
to `D`.

## 3. End-to-end state flow

```text
demonstration/query ids D_t
          │ embedding + parameter-free LayerNorm
          ▼
       H_0 [B,N,D]
          │
          │ apply the same BDH block L times
          │ each depth reads and optionally writes its own M_l
          ▼
 latest hidden sequence + contextual CQ state M = {M_0 ... M_(L-1)}
          │
          ├── last position becomes one-position latent workspace H [B,1,D]
          │        │
          │        └── shared model recurrently applied R times
          │            (memory writes may be enabled or frozen)
          │
          └── output projection → first answer-token logits
                                      │
                                      └── autoregressive answer generation
```

There are two recurrences:

- **depth recurrence**: the same block is applied `L` times inside every model
  pass, with a distinct fast-weight state for each depth;
- **reasoning recurrence**: the entire `L`-depth model is repeatedly applied to
  one continuous position `R` times.

Consequently one thought step performs `L` block applications, not one.

## 4. One shared BDH block, line by line

Let `X` have shape `[B,N,D]`. `src/model.rs::BdhBlock::forward` computes the
following.

### 4.1 Positive high-dimensional features

```text
G = ReLU(X W_qk)                         [B,N,H*Q]
G = reshape/permute(G)                   [B,H,N,Q]
Q_rot = partial_rope(G)
K_rot = partial_rope(G)
```

The same projection produces query, key, and later multiplicative gate. ReLU
makes this large feature space non-negative. The unrotated `G` is kept for the
gate because rotation introduces signed components.

The upstream-compatible default rotates the first half of each head. An
explicit `rotary_dim` may select a narrower even prefix (v2 uses 64 of 768
features); the remainder stays semantic and non-negative. RoPE consumes one
position id per batch row and token rather than assuming all rows share a
cursor.

### 4.2 Read current and past context

For the current chunk:

```text
A_current = tril(Q_rot K_rot^T, diagonal = -1) V   [B,H,N,D]
```

The strict lower triangle means position `i` sees positions `< i`, not itself.
There is no softmax and no `1/sqrt(Q)` scaling.

History is represented by one associative matrix per depth:

```text
M_l = Σ_old K_old^T V_old                 [B,H,Q,D]
A_past = Q_rot M_l                        [B,H,N,D]
A = LayerNorm(A_current + A_past)
```

This is exact for the reconstruction's unnormalized linear attention because
matrix multiplication is associative:

```text
Q (K^T V) = (Q K^T) V
```

The amount of stored contextual state does not grow with the number of old
tokens. Its size grows with `B*L*H*Q*D` instead.

`BdhForwardOptions::document_starts` can mark different reset positions in
each batch row. A token after a reset cannot attend to earlier-document tokens
inside the chunk or read the prior CQ row. RoPE restarts at zero for that row,
and the stored write retains only the final document. This is how stateful
training keeps one efficient batched forward without leaking documents across
lanes.

### 4.3 Lift, gate, and communicate back down

Each head has a learned `D -> Q` lift `W_up,h`:

```text
Z_h = ReLU(A_h W_up,h * G_h)              [B,H,N,Q]
Y = LayerNorm(concat_h(Z_h) W_out)        [B,N,D]
```

This is the high-dimensional computation / low-dimensional communication
pattern inherited from BDH. The public block has three dominant matrices:
`W_qk`, all `W_up,h`, and `W_out`, each containing approximately
`D*(H*Q)` parameters. That mirrors the original paper's approximate `3nd`
parameter structure.

### 4.4 Write the associative state

The current chunk proposes:

```text
ΔM_l = K_rot^T V                           [B,H,Q,D]
M_l' = M_l + ΔM_l                          legacy additive update
ρ_(h,q) = sigmoid(raw_ρ_(h,q))
M_l' = ρ ⊙ M_l + ΔM_l                      optional v2 learned decay
M_l' = M_l                                 when update_memory = false
```

The block always computes its local write because it is also useful for a
fresh state; the model decides whether to commit it. Every recurrent depth has
its own `M_l`, even though all depths share `W_qk`, `W_up`, and `W_out`.

`BdhForwardOptions::valid_sequence_length` supports a physically padded token
pass. If the physical length is `P` and only the first `N <= P` positions are
real, the implementation zeros the trailing input embeddings once before the
shared recurrent block. All block projections are bias-free; zero Q/K gates,
the causal residual, and the memory read therefore keep that tail zero through
every depth. Consequently padding contributes exactly zero to `ΔM_l`.
`Memory.position_offsets[row]` advances by `N`, while logits and
`Memory.embeds` retain physical length `P`. Stateful validation uses bounded
physical buckets in `{16, 32, 64, 128, 256}`; production training normally
uses one unpadded `[B,256]` call.

### 4.5 Recurrent residual

The legacy/public-reconstruction path uses:

```text
X_(l+1) = X_l + Y_l
```

and applies one final parameter-free LayerNorm after all `L` applications.
Architecture v2 enables `normalize_each_depth`, changing the identity path to

```text
X_(l+1) = LayerNorm(X_l + Y_l)
```

and likewise applies parameter-free LayerNorm to the Attention Residual / MHAR
readout before it becomes the next depth's narrow state. The optional
Attention Residual replaces identity addition with a learned mixture of the
seed plus true block deltas `[X_0, Y_0, Y_1, ...]`. Multi-Head Attention
Residual divides `D` into contiguous feature subspaces and gives each one an
independent softmax distribution over saved depths. Full-width RMSNorm is
applied before the split, the learned query is zero-initialized, and H=1
exactly recovers the older single-query router. LayerNorm after the readout
does not change that zero-query uniform routing distribution. These mechanisms
are experimental extensions, not disclosed BDH-CQ components.

### 4.6 Architecture-v2 wide delta state

V2 makes `[B,H,N,Q]` a genuine recurrent computational state while retaining
`D` as the communication channel between depths. With `S_prev` absent, the
first candidate is accepted whole—there is no artificial 80% retention of
zero:

```text
G_l = ReLU(W_qk X_l + alpha ⊙ RMSNorm(S_prev))
Z_l = BDH_neuron_update(G_l, X_l, CQ_l)

if S_prev is None:
    DeltaS_l = Z_l
    S_l = Z_l
else:
    u_(b,h,n,q) = sigmoid(W_u(X_l)_(b,n,h) + raw_u_(h,q))
    DeltaS_l = u_l ⊙ (Z_l - S_prev)
    S_l = S_prev + DeltaS_l

Y_l = LayerNorm(W_out concat(DeltaS_l))
X_(l+1) = LayerNorm(X_l + Y_l)             # v2 identity residual
```

`raw_u` is `[H,Q]`, while the cheap input-dependent projection remains only
`D -> H` and broadcasts over `Q`. Thus neuron coordinates in one head may
learn different baseline timescales without a `D -> H*Q` gate matrix.
Injection is also per-neuron:
`alpha = tanh(raw_alpha)`, `raw_alpha: [H,Q]`. Production v2 initializes
`alpha=0.05`: non-zero enough for a live direct path, yet small relative to
`W_qk X`. There is deliberately no `(H*Q)^2` transition.

The distinction between `S_l` and `DeltaS_l` is essential. `S_l` is the full
wide state used at the next recurrent application. Only `DeltaS_l` enters
`proj_out`, so additive residuals and MHAR never re-project and re-add all old
wide information. MHAR history therefore remains `[X0, Y0, Y1, ...]`, with
each `Y_l` representing new information from that application.

For ordinary token calls the wide state starts empty and is discarded after
the shared block has run `L` times; it never crosses token chunks. For
`Stage::Think(R)`, where `N=1`, [`LatentWorkspace`](../src/model.rs) carries
`[B,H,1,Q]` from reasoning iteration `r` to `r+1` with gradients intact. A new
`Think` chain starts with an empty workspace, and the workspace is never
stored in `Memory.fast_weights`.

## 5. Contextual memory versus latent workspace

The distinction is central:

| State | Rust representation | Shape | Purpose |
|---|---|---|---|
| contextual CQ state `M` | `Memory.fast_weights` | `L × [B,H,Q,D]` | compressed facts/associations from ingested context |
| latest hidden sequence | `Memory.embeds` | `[B,N,D]` | output of the latest pass |
| latent workspace `H` | last position sliced from `Memory.embeds` | `[B,1,D]` | continuously transformed reasoning state |
| latent-only wide workspace `S` | `LatentWorkspace` | `[B,H,1,Q]` | full neuron state inside one `Think(R)` chain |

`Stage::Think(R)` does not sample hidden tokens. It slices the last hidden
position, feeds it back as a continuous embedding, and runs the model `R`
times while carrying its separate wide workspace. If latent memory writes are
enabled, thinking also modifies contextual `M`; if disabled, each iteration
can still read `M`, while all CQ matrices remain frozen. The wide workspace is
independent of that write policy.

`Memory.position_offsets` contains one RoPE cursor per batch row. It advances
for both real tokens and thought positions, resets independently at document
boundaries, and is preserved across chunks. Physical padding declared through
`valid_sequence_length` does not advance a cursor.

## 6. How a stage program executes

The most instructive wrapper call is:

```rust,ignore
[
    Stage::Tokens(prompt),
    Stage::Think(8),
    Stage::Tokens(answer),
]
```

It means:

1. embed the complete prompt and update contextual state;
2. take the prompt's final hidden position and transform it continuously eight
   times;
3. teacher-force the answer tokens using the resulting state;
4. return logits from the final answer segment and the final recurrent memory.

The wrapper accepts several token/embedding/thought segments and supports a
per-stage memory-write override. Protocol errors—thinking before ingestion,
batch changes, or an override list of the wrong length—are returned as
`BdhError`.

## 7. Training objectives

`ReasoningWrapper::forward(... compute_loss = true)` combines:

- **latent supervision**: every thought iteration predicts the first token of
  the following discrete segment;
- **answer next-token prediction**: answer position `i` predicts `i+1`.

`icq::train_loss` adds a third term:

- **prompt next-token prediction**, excluding positions whose target is the
  `<input>` marker.

The ARC loss optionally applies the upstream 14-class weights, downweighting
black background and upweighting colors/structural answer tokens. This
objective is the public reconstruction's training design, not an equation
specified in the BDH-CQ paper.

Burn's autodiff graph crosses prompt ingestion, latent iterations, answer
teacher forcing, and the persistent tensor state. `examples/train_tiny_icq.rs`
shows the complete `loss -> backward -> AdamW step` path.

The core does not require the reasoning wrapper. For ordinary language
modeling, pass `[B,N]` ids directly to `Bdh`, reshape `[B,N,V]` logits for
next-token cross-entropy, and optimize normally. `examples/train_tiny_bytes.rs`
is the compact counterpart of upstream `train_enwik8.py` and demonstrates
recurrent byte generation without downloading enwik8.

## 8. Autoregressive generation

Generation first executes caller-provided stages—usually `Think(R)` with an
already-ingested prompt memory. It projects the final hidden position to the
first answer-token logits, then repeatedly:

1. greedily selects a token when `temperature == 0`, otherwise performs
   thresholded top-k sampling;
2. stops if the token is the configured stop marker;
3. looks up that token's raw embedding;
4. passes the one-position embedding through the continuous-input branch;
5. updates memory and obtains the next logits.

Step 3 deliberately does not apply token-input LayerNorm. This surprising
detail is preserved from the public Python generation loop.

## 9. ARC-style in-context-query protocol

The vocabulary contains ten colors plus:

| id | marker |
|---:|---|
| 10 | row separator |
| 11 | input-grid start |
| 12 | output-grid start |
| 13 | end of output |

A complete prompt is three serialized demonstration input/output pairs plus a
held-out query input. The public helper ingests it in chunks of at most 128
positions so the same fixed-size recurrent state crosses chunk boundaries.

The four generators are deliberately small architecture probes:

- propagation extends a colored bar;
- copying transfers a motif from its source to gray anchors;
- ordering sorts colored vertical bars by height;
- nesting recolors the region inside concentric frames.

They share task-specific layout parameters between easy demonstrations and a
harder held-out query. They are not the full undisclosed training curriculum
used by Pathway's paper experiments.

## 10. Memory and compute intuition

The fast state contains `B*L*(H*Q)*D` scalars. At float32:

```text
bytes = 4 * B * L * (H*Q) * D
```

For the upstream core defaults (`B=1`, `L=8`, `H*Q=32768`, `D=512`) this is
512 MiB. The ARC script's smaller shape (`L=4`, `H*Q=2048`, `D=384`) is 12 MiB
per batch item. Training needs additional activation, gradient, and optimizer
storage.

Within a chunk, the explicit causal similarity matrix is `[B,H,N,N]`, so this
implementation still has quadratic work/memory in the **current chunk
length**. The stored history, however, is fixed-size rather than an ever-growing
KV cache. Chunk size is therefore both a speed/memory knob and, because
LayerNorm and strictly causal local attention are involved, part of observable
execution behavior.

## 11. Where to put breakpoints

- `BdhBlock::forward`: inspect `gates`, `similarity`, `aggregate`, `lifted`, and
  `memory_write` to understand one depth.
- `Bdh::forward`: watch `previous_weights` and `next_weights` to see independent
  depth state and weight tying.
- `ReasoningWrapper::forward`: watch `latent` and the `update` flag during a
  `Think` stage.
- `icq::ingest_hiddens`: see memory cross chunk boundaries.
- `icq::train_loss`: see the three loss sources assembled.

Run `cargo doc --offline --no-deps` for browsable API documentation, and use
`RUST_BACKTRACE=1` if experimenting with low-level invalid tensor shapes.
