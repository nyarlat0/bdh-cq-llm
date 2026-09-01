#!/usr/bin/env python3
"""Benchmark H*Q candidates on the real packed v2 corpus.

Each candidate runs one complete 256-sequence work block with CQ enabled from
the first token. Configs and checkpoints live under /tmp and are never confused
with production. Existing output directories are refused, not overwritten.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path


WIDTHS = (4096, 5120, 6144)


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "0"
    base = json.loads(Path("configs/rx6700-v2.json").read_text(encoding="utf-8"))
    subprocess.run(["cargo", "build", "--release", "--bin", "train_llm"], check=True)
    rows = []
    for width in WIDTHS:
        run_dir = Path(f"/tmp/bdh-cq-v2-width-{width}")
        config_path = Path(f"/tmp/bdh-cq-v2-width-{width}.json")
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; remove it explicitly before rerunning")
        config = json.loads(json.dumps(base))
        config["run_dir"] = str(run_dir)
        config["model"]["dim_qk_heads"] = width
        config["memory"]["stateful_after_tokens"] = 0
        config["checkpoint_every_steps"] = 1_000_000
        config["validation_every_steps"] = 1_000_000
        config["log_every_steps"] = 1
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "./target/release/train_llm",
                "--config",
                str(config_path),
                "--device",
                device,
                "--max-steps",
                "4",
            ],
            check=True,
        )
        events = [
            json.loads(line)
            for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        speeds = [event["tokens_per_second"] for event in events if event["event"] == "train"]
        losses = [event["loss"] for event in events if event["event"] == "train"]
        # Ignore the compilation/autotune-heavy first update. Keeping the
        # production GA=16 and one-chunk TBPTT is essential: a cheaper benchmark could
        # fit even when the actual activation graph would OOM.
        rows.append((width, statistics.median(speeds[1:]), statistics.mean(losses[-2:])))

    print("H*Q    median tok/s    last-2 train loss")
    for width, speed, loss in rows:
        print(f"{width:<6} {speed:>12.0f} {loss:>20.5f}")
    print("Choose 6144 only if it fits with margin and is not disproportionately slower.")


if __name__ == "__main__":
    main()
