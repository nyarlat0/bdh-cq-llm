#!/usr/bin/env bash
# Compare RoPE widths without changing any other architecture or budget knob.

set -eu

device="${1:-0}"
steps=1220

if [ ! -f artifacts/tokenizer-v2-24576.json ]; then
    echo "Missing artifacts/tokenizer-v2-24576.json; run ./scripts/prepare_v2_data.sh first." >&2
    exit 1
fi

for shard in fineweb2_hq ficbook ru_classic; do
    if [ ! -f "datasets/packed/rx6700-v2-24576/$shard.tokens" ]; then
        echo "Missing packed shard $shard; run ./scripts/prepare_v2_data.sh first." >&2
        exit 1
    fi
done

cargo build --release --bin train_llm

for width in 64 192 384; do
    run_dir="runs/rx6700-v2-rope-$width"
    if [ -e "$run_dir" ]; then
        echo "$run_dir already exists; refusing to resume a fixed-budget comparison." >&2
        exit 1
    fi
    ./target/release/train_llm \
        --config "configs/rx6700-v2-rope-$width.json" \
        --device "$device" \
        --max-steps "$steps"
done

python3 scripts/summarize_v2_rope_sweep.py
