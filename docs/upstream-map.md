# Upstream and paper crosswalk

This document answers two different fidelity questions: “where did this Rust
code come from?” and “which parts are actually disclosed by the papers?”

## Pinned sources

- Public reconstruction: [`lucidrains/bdh-cq`](https://github.com/lucidrains/bdh-cq),
  commit `720f0c62844af2d14c99750faecfa82f05f23ae1`, 2026-08-20.
- BDH-CQ paper: [*BDH-CQ: A Brain-inspired Architecture for General-Purpose
  AI*](https://arxiv.org/abs/2608.09888), arXiv:2608.09888.
- Original architecture paper: [*Dragon Hatchling: The Missing Link between
  the Transformer and Models of the Brain*](https://arxiv.org/abs/2509.26507),
  arXiv:2509.26507.
- Original reference implementation:
  [`pathwaycom/bdh`](https://github.com/pathwaycom/bdh).

## Paper equations to Rust concepts

| Research concept | Rust location | Interpretation in this port |
|---|---|---|
| `S_t = U_theta(S_(t-1), D_t)` | `Bdh::forward`, `Memory` | additive per-depth `M <- M + K^T V` while processing a token chunk |
| initial workspace `H_0` | `Memory.embeds`, `Stage::Think` | final ingested hidden position |
| `H_(r+1) = F_theta(H_r, S_K)` | `ReasoningWrapper::forward` | feed one continuous position through the complete shared-depth model |
| decode from final workspace | `project_logits`, `generate` | vocabulary projection followed by autoregression |
| positive high-dimensional activations | `BdhBlock::to_qk`, ReLU | shared Q/K/gate features of width `H*Q` |
| low-rank communication | `proj_up`, `proj_out` | `D -> Q` per head, then `H*Q -> D` |
| layerwise fixed state | `Memory.fast_weights` | one `[B,H,Q,D]` tensor for each recurrent depth |
| shared parameters across depth | the single `BdhBlock` field | the same parameters are applied `depth` times |

The first four rows are system-level concepts published by the BDH-CQ paper.
The exact formulas in the third column are choices made by the public
reconstruction; the paper explicitly does not disclose them.

## Python-to-Rust source map

| Python source | Rust counterpart | Coverage |
|---|---|---|
| `bdh_cq/bdh_cq.py::BDHBlock` | `src/model.rs::BdhBlock` | Q/K/gate projection, partial RoPE, causal attention, fast-state read/write, low-rank projections |
| `bdh_cq/bdh_cq.py::BDH` | `src/model.rs::Bdh` | embedding, shared recurrent depth, per-depth memories, logits |
| `AttentionResidual` and depth-bias helper | `AttentionResidual`, `compute_attn_residual_depth_bias` | optional history mixing and upstream indexing rule |
| `BDHReasoningWrapper.forward` | `ReasoningWrapper::forward` | token/embed/latent interleaving, write policies, losses |
| `BDHReasoningWrapper.generate` | `ReasoningWrapper::generate` | greedy or top-k temperature sampling and raw-embedding feedback |
| `bdh_cq/tasks.py` | `src/tasks.rs` | all four task families, level control, persistent parameters |
| `bdh_cq/icq.py` codec | `src/icq.rs` constants and codec functions | 14-token serialization, ragged row padding, prompt/answer construction |
| `ingest`, `ingest_hiddens` | same Rust function names | recurrent chunking and hidden collection |
| `train`, wrapper loss | `icq::train_loss` | prompt, latent, and answer losses; optimizer step lives in an example |
| `generate_answer`, `solve`, `cell_stats` | same Rust function names | complete inference helpers |
| `figure7.py` | `examples/train_tiny_icq.rs` | compact mechanics example rather than the long sweep/reporting harness |
| `train_enwik8.py` | `examples/train_tiny_bytes.rs` | same direct next-byte model/loss and recurrent sampling mechanics, with an embedded toy corpus instead of downloading enwik8 |

The architecture and reusable algorithms are ported. Upstream experiment
orchestration—external dataset acquisition, long sweeps, checkpoint naming,
experiment tracking, and plotting—is intentionally represented by small
examples rather than copied as an environment-specific training application.

## Behaviors deliberately preserved

- A single `BdhBlock` owns all learned depth parameters.
- Each depth nevertheless owns an independent recurrent matrix.
- Current-chunk attention is unnormalized and strictly causal; the diagonal is
  zero.
- The same positive projection supplies Q, K, and the later gate.
- RoPE applies to only half of each head's features.
- Memory is additive, with no decay or normalization at write time.
- Thought steps can independently enable or freeze memory writes.
- Latent steps are supervised against the first token of the next discrete
  segment.
- Autoregressive feedback uses raw token embeddings through the continuous
  input path.
- Generation is batch-one because every chosen token is read synchronously on
  the host, matching the public wrapper's restriction.
- The copying generator preserves upstream's unusual source-block placement,
  and propagation preserves its Python/NumPy negative-index wraparound where
  it is part of generated behavior.

## Intentional Rust differences

- Python's “integer tensor or float tensor” dispatch is an explicit
  `ModelInput`/`Stage` enum.
- Tuple/dictionary memory is a named `Memory` struct with validated batch and
  depth compatibility.
- Recoverable configuration, protocol, generation, and grid failures return
  `BdhError` instead of relying on assertions.
- Attention-history vectors are passed and returned by ownership, instead of
  mutating a caller-owned Python list.
- Generated token sampling transfers a tiny logit vector to the CPU. This is
  portable across Burn backends, but not optimized for high-throughput
  serving.
- `Grid` rejects empty dimensions. The Python decoder can construct shape
  `(1, 0)` for malformed output; Rust reports that case as `InvalidGrid`.
- Rust and NumPy random generators are not stream-compatible, so an equal seed
  means deterministic Rust data, not identical grids across languages.
- Burn and PyTorch default initializers are not guaranteed to produce
  numerically identical weights. Structural behavior, protocols, formulas,
  and shapes—not checkpoint bit parity—are the fidelity target.

## Features that should not be attributed to the disclosed BDH-CQ model

The public repository is explicitly a work in progress. In particular, its
precise fast-weight update, partial RoPE, latent-step loss, raw-embedding
generation detail, and optional Attention Residual module are public
reconstruction choices. The paper also describes a larger evaluated system
with data transformations, candidate generation, ranking, and continual
operation whose proprietary details are insufficient for an independent exact
implementation.

This separation is why the crate and documentation consistently use phrases
such as “public reconstruction” rather than claiming to reproduce Pathway's
reported 150M-parameter system or benchmark results.
