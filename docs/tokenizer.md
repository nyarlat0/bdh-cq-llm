# Russian pretraining tokenizer

This document explains both the design and the exact path from the three data
sources to `tokenizer.json`. The implementation is split between
[`src/tokenizer.rs`](../src/tokenizer.rs), which defines and trains BPE, and
[`scripts/stream_tokenizer_corpus.py`](../scripts/stream_tokenizer_corpus.py),
which only reads datasets and streams UTF-8 documents.

## Corpus plan versus tokenizer sample

The first model run has a budget of one billion **resulting tokenizer tokens**:

| Source | Model tokens | Weight | Tokenizer's default raw-byte sample |
|---|---:|---:|---:|
| `epfml/FineWeb2-HQ`, `rus_Cyrl` | 650M | 65% | 650MB |
| `IlyaGusev/ficbook` | 300M | 30% | 300MB |
| `Imperius/ru-classic` | 50M | 5% | 50MB |

The final column is intentionally measured in UTF-8 bytes. Before a tokenizer
exists, “take N tokens” is circular: token boundaries are exactly what is being
learned. The default one-gigabyte stratified sample provides many more BPE pair
observations than the 32K vocabulary needs without staging the whole 1B-token
training stream on disk. Change it with `--sample-bytes`, but retain 65/30/5.

Byte weights only approximate token weights because Russian and web markup have
different bytes-per-token ratios. The **model data packer**, not tokenizer
training, must encode each source and stop at exactly 650M, 300M and 50M token
IDs. Likewise, Ficbook's 50M first pass followed by 250M in the second pass is a
model curriculum decision. BPE is an order-independent frequency learner and
therefore sees the combined 30% Ficbook weight.

## Why byte-level BPE

The vocabulary is byte-level BPE with the GPT-2 whitespace/word regex:

1. Every one of the 256 possible bytes is placed in the initial alphabet.
2. Russian text, Latin fragments, emoji, malformed-looking web strings and any
   future UTF-8 input can therefore be encoded without an unknown token.
3. BPE repeatedly merges frequent adjacent symbols. On this mixture it learns
   Cyrillic character sequences, endings, stems, common words and punctuation
   patterns while retaining byte fallback for rare strings.

There is no NFC/NFKC normalization, case folding or whitespace cleanup. This is
important for studying language as it appears in the datasets: `е` and `ё`
stay different, multiple spaces survive, and decode(encode(text)) equals text.
The maximum learned token length is 64 bytes so that a long URL or repeated web
identifier cannot consume a vocabulary entry. Input itself is never truncated
by this limit.

The default vocabulary is 32,768 entries. For a small language model this is a
reasonable compromise between Russian compression and the parameter cost of
the embedding/output matrices. Do not increase it solely because more corpus is
available; compare held-out fertility (tokens per character/word), compression,
and downstream validation loss first.

## Reserved IDs

These IDs are an ABI shared by the tokenizer, dataset packer and checkpoints:

| ID | Token | Intended use |
|---:|---|---|
| 0 | `<\|pad\|>` | Padding; mask it out of loss and attention |
| 1 | `<\|bos\|>` | Optional beginning of an independently sampled sequence |
| 2 | `<\|eos\|>` | End of a generated sample/conversation |
| 3 | `<\|doc\|>` | Boundary between packed pretraining documents |
| 4 | `<\|system\|>` | Future chat/instruction role |
| 5 | `<\|user\|>` | Future chat/instruction role |
| 6 | `<\|assistant\|>` | Future chat/instruction role |
| 7 | `<\|eot\|>` | End of one chat turn |

There is no `<unk>`. The byte alphabet starts at ID 8, followed by learned BPE
merges. `train_tokenizer` validates the reserved IDs and a lossless Russian/emoji
round trip before saving an artifact.

`<|doc|>` is not inserted into text while learning BPE. The later data packer
encodes a document with `add_special_tokens = false`, then explicitly inserts
ID 3. This makes document boundaries visible to the language model without
polluting ordinary text statistics.

## What the sampler reads

- FineWeb2-HQ is loaded as an iterable dataset with config `rus_Cyrl`. Only the
  `text` column is projected, which avoids transferring its large auxiliary
  columns. A seeded streaming shuffle prevents the tokenizer from seeing only
  the first shards.
- Local Ficbook Parquet shards are shuffled deterministically and read in
  bounded PyArrow batches (without converting all 64GB to Arrow). Only
  non-empty `parts[*].clean_text` bodies are emitted; story/chapter titles,
  description, tags and rating do not enter the BPE corpus.
  `--ficbook-part-field text` selects the raw body alternative. No row is
  excluded because of rating, tags, vocabulary or semantic content.
- `ru-classic.txt` is read from seeded shuffled 4MiB windows instead of taking
  only the start of the 869MB local file.

Each emitted sequence is at most 1MiB, preventing a single pathological record
from becoming a large allocation. It may still contain arbitrary newlines. The
binary framing protocol carries a source ID and a 64-bit byte length, so no text
escaping or line-based corruption occurs.

The checked-in `artifacts/tokenizer.json` predates the body-only adapter: its
Ficbook sample also contained metadata, as recorded by
`ficbook_metadata_included: true` in its manifest. That only influenced BPE
merge statistics; the production LM token shards contain bodies only. Running
the tokenizer command again now uses body-only input and intentionally creates
a new, checkpoint-incompatible tokenizer version.

## Train it

Install the Python reader dependencies once (preferably in a virtualenv):

```console
python3 -m pip install -r scripts/requirements-tokenizer.txt
```

Then run the release build from the repository root:

```console
cargo run --release --bin train_tokenizer -- \
  --sample-bytes 1GB \
  --output artifacts/tokenizer.json \
  --manifest artifacts/tokenizer.manifest.json
```

FineWeb remains remote and streamed; no 1.2TB checkout is created. The defaults
expect the already downloaded `datasets/ficbook/*.parquet` shards and
`datasets/ru-classic.txt`.

For a fast dependency-free plumbing check:

```console
cargo run --release --bin train_tokenizer -- \
  --smoke-fixture --sample-bytes 1MB \
  --vocab-size 2048 \
  --output /tmp/bdh-cq-tokenizer.json \
  --manifest /tmp/bdh-cq-tokenizer.manifest.json
```

The normal run writes two files:

- `tokenizer.json` is the self-contained Hugging Face Tokenizers artifact used
  by Rust or Python inference/data tooling.
- `tokenizer.manifest.json` records hyperparameters, source locations, planned
  and observed byte counts, the random seed and special IDs.

Inspect the finished artifact and its exact BPE pieces with:

```console
cargo run --release --example tokenizer_roundtrip -- \
  artifacts/tokenizer.json "Привет! Проверяем токенизацию русской речи."
```

The default Hub revision is pinned to
`c0c06e94fd3a44ae9e802b2b0fc533817601eb5e` (resolved on 2026-08-25).
Pass a different `--fineweb-revision <commit-sha>` deliberately when updating
the corpus; using moving `main` can alter the stream even with the same seed.

## Acceptance checks before freezing the ABI

Once the real artifact is trained, freeze it only after:

1. exact round-trip tests on Russian, Latin, emoji, combining marks, whitespace
   and all eight literal special-token strings;
2. held-out fertility reports for each source separately, especially Ficbook
   dialogue and Russian morphology;
3. frequency counts for IDs 0--7 after packing, ensuring ordinary source text
   cannot accidentally be treated as control structure;
4. a small model overfit run confirming that padding is masked and document
   boundaries are inserted correctly; and
5. recording a pinned dataset revision plus hashes of the local data manifest.

Changing vocabulary entries or reserved IDs after model training begins makes
old packed data and checkpoints incompatible. Treat `tokenizer.json` as part of
the checkpoint format, not as a replaceable preprocessing convenience.
