#!/usr/bin/env bash
# Build the body-only 24,576-token tokenizer and pack all production shards.
#
# This script intentionally passes no metadata/content-filter flags. FineWeb
# contributes only its `text` column, Ficbook only `parts[].clean_text`, and
# ru-classic its text-file contents. The packer inserts only the structural
# <|doc|> boundary required to reset CQ memory.

set -eu

# Resolve every relative path from the repository root, even when the wrapper
# is invoked through an absolute path or from another working directory.
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

# Hugging Face otherwise defaults to ~/.cache, which may be read-only in a
# managed workspace and also makes the supposedly reproducible command depend
# on per-user state. Keep downloads beside the ignored raw datasets instead.
HF_HOME="${HF_HOME:-datasets/.cache/huggingface}"
export HF_HOME
mkdir -p "$HF_HOME"

# With no argument, keep a project-local, ignored environment so the command
# survives reboots and works exactly as printed in the documentation. An
# explicit Python path remains available for users who manage their own env.
python_bin="${1:-.venv-tokenizer/bin/python}"
managed_environment=false
if [ "$python_bin" = ".venv-tokenizer/bin/python" ]; then
    managed_environment=true
fi

if [ ! -x "$python_bin" ]; then
    if [ "$managed_environment" = true ]; then
        echo "Creating persistent tokenizer environment in .venv-tokenizer ..." >&2
        python3 -m venv .venv-tokenizer
    else
        echo "Python environment is missing or not executable: $python_bin" >&2
        exit 1
    fi
fi

if ! "$python_bin" -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    if [ "$managed_environment" = true ]; then
        echo "Installing tokenizer data-reader dependencies ..." >&2
        "$python_bin" -m pip install -r scripts/requirements-tokenizer.txt
    else
        echo "$python_bin is missing datasets/pyarrow." >&2
        echo "Install scripts/requirements-tokenizer.txt into that environment." >&2
        exit 1
    fi
fi

if [ ! -d datasets/ficbook ]; then
    echo "datasets/ficbook is missing" >&2
    exit 1
fi

if [ ! -f datasets/ru-classic.txt ]; then
    echo "datasets/ru-classic.txt is missing" >&2
    exit 1
fi

tokenizer=artifacts/tokenizer-v2-24576.json
manifest=artifacts/tokenizer-v2-24576.manifest.json

if [ -f "$tokenizer" ] && [ -f "$manifest" ]; then
    echo "Reusing existing v2 tokenizer ABI: $tokenizer" >&2
elif [ -e "$tokenizer" ] || [ -e "$manifest" ]; then
    echo "Incomplete tokenizer ABI: $tokenizer and $manifest must either both exist or both be absent." >&2
    echo "Inspect the partial result and move it aside before retrying." >&2
    exit 1
else
    cargo run --release --bin train_tokenizer -- \
        --python "$python_bin" \
        --sample-bytes 1GB \
        --vocab-size 24576 \
        --ficbook-part-field clean_text \
        --output "$tokenizer" \
        --manifest "$manifest"
fi

# The packer validates and reuses each complete shard. If preparation was
# interrupted, rerunning this script resumes at the first absent shard instead
# of deleting the valid work already completed.
cargo run --release --bin pack_pretraining_data -- \
    --config configs/rx6700-v2.json \
    --python "$python_bin"

echo "v2 tokenizer and packed corpora are ready. Do not start production before the pilot report."
