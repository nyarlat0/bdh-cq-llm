#!/usr/bin/env python3
"""Fresh equal-token FP32 learning pilots at previously measured fastest batches.

No production config/checkpoint is changed. Wall times include initialization,
validation and earlier saves, measured independently for each model process.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from benchmark_v100 import command, read_gpu
from benchmark_v100_sizes import make_config, parse_result


CASES = ((512, 8), (640, 8), (768, 8), (1024, 4))


def pilot_config(base, dim, batch, run_dir, interval):
    config = make_config(base, dim, batch, run_dir)
    config.update(validation_every_steps=interval, checkpoint_every_steps=interval,
                  validation_batches=384, log_every_steps=16)
    return config


def write_summary(output, rows):
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# V100 size learning pilots", "",
             "Fresh FP32 runs, same token budget/LR, CQ active from token zero. "
             "Only completed finite runs with final validation are eligible. "
             "A single seed and shared LR do not establish optimal quality for each width.", "",
             "| D | Batch | Params | Completed | Eligible | Best loss | Last loss | tok/s | Wall hours |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key)) for key in
                     ("dim", "batch", "parameters", "completed", "eligible", "best_loss",
                      "last_loss", "median_tok_s", "wall_hours")) + " |")
    lines += ["", "Use validation.csv for loss versus tokens and wall_seconds; "
              "per-source BPB and memoryless/stateful metrics remain in each train.jsonl. "
              "Wall time includes initialization, validation and preceding checkpoints. "
              "Compare first threshold crossings in CSV, not just final loss, for time-to-quality."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/v100-v2.json"))
    parser.add_argument("--updates", type=int, default=3072)
    parser.add_argument("--validation-every", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.validation_every < 16 or args.validation_every % 16
            or args.updates < args.validation_every or args.updates % args.validation_every):
        parser.error("validation interval must be a multiple of 16; updates a positive multiple of interval")
    base = json.loads(args.config.read_text())
    if base["sequence_length"] != 256 or base["memory"]["chunks_per_detach"] != 2:
        parser.error("expected context=256 and TBPTT=2")
    if base["model"]["rotary_dim"] * 2 * base["model"]["heads"] != base["model"]["dim_qk_heads"]:
        parser.error("base must use RoPE=Q/2")
    # Fail early on missing input files without rebuilding or touching datasets.
    if not args.dry_run:
        required = [Path(base["tokenizer"])] + [Path(base["packed_dir"]) / f"{s}.tokens"
                                               for s in base["sources"]]
        for path in required:
            if not path.is_file():
                parser.error(f"missing input: {path}")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or Path("runs") / f"v100-size-pilots-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for dim, batch in CASES:
        label = f"fp32-d{dim}-b{batch}"
        config = pilot_config(base, dim, batch, output / label, args.validation_every)
        path = output / f"{label}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        cases.append((dim, batch, label, path))
    print(f"Four pilots, {args.updates * 16384:,} tokens each; {output}", flush=True)
    if args.dry_run:
        return
    nvcc = command(["nvcc", "--version"], capture_output=True).stdout
    if not re.search(r"release 12\.9\b", nvcc):
        raise SystemExit("Select CUDA Toolkit 12.9, including NVRTC/libraries.")
    uuid, gpu = read_gpu(args.device)
    env = dict(os.environ, CUDARC_CUDA_VERSION="12090", CUDA_VISIBLE_DEVICES=uuid)
    (output / "hardware.txt").write_text(gpu + "\n" + nvcc)
    command(["cargo", "build", "--release", "--locked", "--no-default-features", "--features", "cuda",
             "--bin", "train_llm"], env=env)
    target = Path(json.loads(command(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                                    capture_output=True, env=env).stdout)["target_directory"])
    binary = output / "train-fp32"
    shutil.copy2(target / "release/train_llm", binary)
    rows = []
    with (output / "validation.csv").open("w", newline="") as curve:
        writer = csv.writer(curve)
        writer.writerow(["dim", "batch", "step", "tokens", "wall_seconds", "memoryless_loss", "stateful_loss"])
        curve.flush()
        for dim, batch, label, path in cases:
            print(f"Starting {label}; log: {output / (label + '.log')}", flush=True)
            started = time.monotonic()
            # Inherit stdout's lines into both the terminal and a retained log.
            # A global STOP file stops the whole sequence, not only this case.
            with (output / f"{label}.log").open("w") as log:
                process = subprocess.Popen([str(binary), "--config", str(path), "--device", "0",
                    "--max-steps", str(args.updates)], env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        if (output / "STOP").exists():
                            (output / label / "STOP").touch()
                        match = re.search(r"validation step (\d+): memoryless ([\d.eE+-]+), stateful Some\(([\d.eE+-]+)\)", line)
                        if match:
                            step, memoryless, stateful = match.groups()
                            writer.writerow([dim, batch, step, int(step) * 16384,
                                             time.monotonic() - started, memoryless, stateful])
                            curve.flush()
                    status = process.wait()
                except KeyboardInterrupt:
                    # Foreground Ctrl-C also reaches the trainer's graceful
                    # handler. Do not start another pilot after user cancellation.
                    print(f"Pilot sequence cancelled; trainer PID {process.pid}; "
                          f"wait for its safe checkpoint in {output / label}.", flush=True)
                    raise SystemExit(130)
            content = (output / f"{label}.log").read_text()
            row = parse_result(content, status, 32)
            events_path = output / label / "train.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
            validation = [e for e in events if e.get("event") == "validation"]
            row["eligible"] &= bool(validation and validation[-1]["step"] == args.updates)
            row.update(dim=dim, batch=batch, wall_hours=(time.monotonic() - started) / 3600,
                       best_loss=min((e["selected_loss"] for e in validation), default=None),
                       last_loss=validation[-1]["selected_loss"] if validation else None)
            rows.append(row)
            write_summary(output, rows)
            # No automatic smaller-batch retry: that would silently change the
            # agreed experiment. STOP/non-finite/OOM all stop the sequence.
            if not row["eligible"] or (output / "STOP").exists():
                raise SystemExit(f"Sequence stopped; inspect {output / 'summary.md'} and {label}.log")
    print(f"Done: {output / 'summary.md'}. No production training started.")


if __name__ == "__main__":
    main()
