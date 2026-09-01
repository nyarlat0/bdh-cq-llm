#!/usr/bin/env bash
# Run four architecture pilots plus an H=1 AttnRes control on GPU 0.

set -eu

device="${1:-0}"
# 1,220 is divisible by the four optimizer updates in a 256-sequence work
# block, so graceful stopping does not overshoot one pilot's token budget.
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

for name in additive attnres state attnres-state attnres-state-h1; do
    run_dir="runs/rx6700-v2-pilot-tbptt1-$name"
    if [ -e "$run_dir" ]; then
        echo "$run_dir already exists; refusing to resume and corrupt a fixed-budget comparison." >&2
        exit 1
    fi
    ./target/release/train_llm \
        --config "configs/rx6700-v2-pilot-$name.json" \
        --device "$device" \
        --max-steps "$steps"
done

python3 scripts/summarize_v2_pilots.py
