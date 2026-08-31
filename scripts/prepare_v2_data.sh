#!/usr/bin/env bash
# Build the body-only 24,576-token tokenizer and pack all production shards.
#
# This script intentionally passes no metadata/content-filter flags. FineWeb
# contributes only its `text` column, Ficbook only `parts[].clean_text`, and
# ru-classic its text-file contents. The packer inserts only the structural
# <|doc|> boundary required to reset CQ memory.

set -eu

python_bin="${1:-/tmp/bdh-cq-tokenizer-venv/bin/python}"

if [ ! -x "$python_bin" ]; then
    echo "Python environment is missing or not executable: $python_bin" >&2
    echo "Create it and install scripts/requirements-tokenizer.txt first." >&2
    exit 1
fi

if [ ! -d datasets/ficbook ]; then
    echo "datasets/ficbook is missing" >&2
    exit 1
fi

if [ ! -f datasets/ru-classic.txt ]; then
    echo "datasets/ru-classic.txt is missing" >&2
    exit 1
fi

for output in \
    artifacts/tokenizer-v2-24576.json \
    artifacts/tokenizer-v2-24576.manifest.json \
    datasets/packed/rx6700-v2-24576; do
    if [ -e "$output" ]; then
        echo "$output already exists; refusing to overwrite a tokenizer/data ABI." >&2
        echo "Move it aside explicitly if you intend to rebuild v2." >&2
        exit 1
    fi
done

cargo run --release --bin train_tokenizer -- \
    --python "$python_bin" \
    --sample-bytes 1GB \
    --vocab-size 24576 \
    --ficbook-part-field clean_text \
    --output artifacts/tokenizer-v2-24576.json \
    --manifest artifacts/tokenizer-v2-24576.manifest.json

cargo run --release --bin pack_pretraining_data -- \
    --config configs/rx6700-v2.json \
    --python "$python_bin"

echo "v2 tokenizer and packed corpora are ready. Do not start production before the pilot report."
