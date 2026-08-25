# BDH-CQ in Rust

This repository is a study-oriented Rust/Burn port of
[`lucidrains/bdh-cq`](https://github.com/lucidrains/bdh-cq). It contains the
tensor-level model, recurrent fast-weight memory, continuous latent-reasoning
wrapper, ARC-style task generators and codec, generation, training losses,
tests, and small runnable examples.

The most important scope note is that this is a port of the **public
reconstruction**, not Pathway's evaluated proprietary model. The
[BDH-CQ paper](https://arxiv.org/abs/2608.09888) gives the system-level
recurrences but explicitly withholds exact dimensions and internal update
rules. The public Python repository fills those gaps using ideas from the
[original BDH paper](https://arxiv.org/abs/2509.26507), plus some experimental
choices of its own. [`docs/upstream-map.md`](docs/upstream-map.md) labels that
boundary precisely.

## Suggested reading order

1. [`docs/architecture.md`](docs/architecture.md) — equations, tensor shapes,
   state flow, training, and memory cost.
2. [`examples/architecture_walkthrough.rs`](examples/architecture_walkthrough.rs)
   — a tiny, inspectable ingest/think/answer pass.
3. [`src/model.rs`](src/model.rs) — the shared BDH block and associative state.
4. [`src/reasoning.rs`](src/reasoning.rs) — continuous latent iterations and
   autoregressive decoding.
5. [`src/tasks.rs`](src/tasks.rs), then [`src/icq.rs`](src/icq.rs) — synthetic
   ARC tasks and their in-context-query protocol.
6. [`docs/upstream-map.md`](docs/upstream-map.md) — Python-to-Rust crosswalk and
   intentional differences.

Every public item is documented, and the dense tensor operations have inline
shape annotations next to the implementation.

## Run it

The default binary uses deliberately small dimensions and the CPU ndarray
backend:

```console
cargo run --offline
cargo run --offline --example architecture_walkthrough
cargo run --offline --example train_tiny_icq -- 10
cargo run --offline --example train_tiny_bytes -- 10
cargo test --offline --all-targets
cargo doc --offline --no-deps --open
```

The training examples are mechanics demonstrations, not reproductions of paper
results. `train_tiny_icq` covers the latent-reasoning objective behind the
figure-7 script; `train_tiny_bytes` covers the ordinary next-byte objective in
`train_enwik8.py` without downloading the 100 MB dataset. Increase their step
counts and replace the CPU backend for real experiments.

## What is implemented

- one positive, high-dimensional Q/K projection shared as the multiplicative
  gate;
- partial rotary position encoding and causal, unnormalized linear attention;
- one fixed-size `K^T V` fast-weight matrix per recurrent depth;
- recurrent-depth weight sharing;
- token ingestion, embedding ingestion, latent “think” stages, latent
  supervision, and answer next-token supervision;
- optional Attention Residual history and its cycle-distance bias;
- greedy or top-k/temperature generation;
- propagation, copying, ordering, and nesting task families;
- the 14-token grid codec and the complete public ARC objective;
- validation tests for state shape, chunk recurrence, memory freezing,
  gradients, codec behavior, generation, and residual-bias indexing.

## Practical warning about the defaults

The upstream core default uses `H*Q = 32,768`, `D = 512`, and depth 8. One
float32 fast-weight state therefore occupies approximately 64 MiB per batch
item per depth, or approximately 512 MiB across all eight depths, before
activations, optimizer state, and gradients. Start with the tiny example
configuration while studying the implementation.

## Fidelity target

The port targets the public Python behavior at commit
`720f0c62844af2d14c99750faecfa82f05f23ae1` (2026-08-20). It preserves subtle
behaviors such as a strictly lower-triangular current-chunk attention mask,
additive memory writes, independent memory per recurrent depth, optional
memory writes during thought, and the raw-embedding feedback path during
generation. Random initialization and random task samples are not bitwise
identical across PyTorch/NumPy and Burn/Rust.

Licensed under MIT; see [`LICENSE`](LICENSE).
