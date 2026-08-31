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
7. [`docs/tokenizer.md`](docs/tokenizer.md) — the Russian byte-level BPE design,
   65/30/5 streaming sample, reserved IDs, and reproducible training command.
8. [`docs/pretraining.md`](docs/pretraining.md) — the exact 1B-token curriculum,
   packed-data format, RX 6700 XT profile, checkpoints, resume, and operations.
9. [`docs/v2-training.md`](docs/v2-training.md) — the redesigned body-heavy
   model, full depth-local neuron state, decaying CQ, 1.05B replay schedule,
   width benchmark, 2×2 pilots, and production gate.

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
cargo run --offline --example tokenizer_roundtrip
cargo test --offline --all-targets
cargo doc --offline --no-deps --open
```

The pretraining tokenizer has its own data adapter and artifact manifest. A
dependency-free smoke run is:

```console
cargo run --release --bin train_tokenizer -- \
  --smoke-fixture --sample-bytes 1MB --vocab-size 2048 \
  --output /tmp/bdh-cq-tokenizer.json \
  --manifest /tmp/bdh-cq-tokenizer.manifest.json
```

See [`docs/tokenizer.md`](docs/tokenizer.md) before running the real 1GB
stratified sample against remote FineWeb and the local Ficbook/classics files.

The actual Russian pretraining path is separate from the tiny examples:

```console
cargo run --release --bin pack_pretraining_data -- \
  --config configs/rx6700.json \
  --python /tmp/bdh-cq-tokenizer-venv/bin/python

cargo run --release --bin train_llm -- --config configs/rx6700.json
```

The production continuation now trains persistent CQ memory without rebuilding
the packed corpus. It imports the last memoryless checkpoint into a separate,
rollback-safe run and then carries memory across 256-token chunks with
1024-token truncated BPTT:

```console
cargo run --release --bin train_llm -- \
  --config configs/rx6700-cq.json \
  --import-checkpoint runs/rx6700-v1/checkpoints/step-000000024000
```

Subsequent resumes use only `--config configs/rx6700-cq.json`. The trainer
resets at packed `<|doc|>` markers and shuffled work-block boundaries.

To inspect the latest saved model as pure text completion without loading the
optimizer or packed corpora:

```console
cargo run --release --bin complete_llm -- --config configs/rx6700-cq.json
```

The REPL inserts the trained `<|doc|>` once, then sends exactly the entered text
and carries CQ memory through generated and user-supplied fragments. It adds no
role labels or hidden separators and supports `/reset`, `/status`, and `/quit`.
This is a base-model probe, not a chat-SFT interface.

It uses Burn/WGPU over Vulkan (Mesa RADV on the tested RX 6700 XT), automatically
resumes its latest checkpoint, and follows the 750M general + 250M Ficbook-focus
curriculum. Read [`docs/pretraining.md`](docs/pretraining.md) before starting a
multi-day run.

The new architecture-v2 run is intentionally separate. Prepare its body-only
24,576-token ABI, benchmark the three body widths and run the four 20M-token
pilots before starting production:

```console
scripts/prepare_v2_data.sh /tmp/bdh-cq-tokenizer-venv/bin/python
python3 scripts/benchmark_v2_widths.py 0
scripts/run_v2_pilots.sh 0
cargo run --release --bin train_llm -- --config configs/rx6700-v2.json
```

These commands do not add chat-role labels or content filters. See
[`docs/v2-training.md`](docs/v2-training.md) for the exact acceptance criteria
and why the production command must be run only after choosing the pilot winner.

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
- optional full positive neuron state carried across recurrent depth, with a
  bounded input-dependent update gate;
- optional learned per-head CQ decay, explicit RoPE width and tied token/LM
  embeddings;
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

Licensed under [The Unlicense](LICENSE)
