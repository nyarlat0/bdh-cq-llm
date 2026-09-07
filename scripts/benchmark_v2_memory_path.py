#!/usr/bin/env python3
"""Benchmark the production CQ/TBPTT2 path without touching its run directory.

The benchmark imports one safe production checkpoint into temporary run
directories. It first measures the normal 4x16 batch partition. Only if that
variant still spills more than the configured GTT threshold does it measure
the mathematically equivalent 2x32 fallback. Temporary checkpoints are deleted
when the script exits; the production run and its latest pointer are read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    name: str
    micro_batch: int
    accumulation: int
    median_tokens_per_second: float
    peak_vram_mib: int
    peak_gtt_mib: int
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rx6700-v2.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--gtt-threshold-mib", type=int, default=512)
    parser.add_argument("--max-idle-vram-mib", type=int, default=1536)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args()


def refuse_while_training() -> None:
    own_pid = os.getpid()
    matches: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if "target/release/train_llm" in command:
            matches.append((int(entry.name), command.strip()))
    if matches:
        details = "\n".join(f"  PID {pid}: {command}" for pid, command in matches)
        raise SystemExit(
            "Refusing to benchmark while another train_llm uses the GPU:\n" + details
        )


def refuse_busy_gpu(max_idle_vram_mib: int, allow_busy: bool) -> None:
    """Keep benchmark numbers from silently including a game or GUI workload."""
    if allow_busy:
        return
    candidates: list[tuple[int, Path]] = []
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        total_path = card / "device/mem_info_vram_total"
        try:
            candidates.append((int(total_path.read_text()), card))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    if not candidates:
        return
    _, card = max(candidates)
    try:
        used_mib = int((card / "device/mem_info_vram_used").read_text()) // (1024 * 1024)
        busy = int((card / "device/gpu_busy_percent").read_text())
    except (FileNotFoundError, PermissionError, ValueError):
        return
    if used_mib > max_idle_vram_mib or busy > 20:
        raise SystemExit(
            f"Refusing to benchmark on busy {card.name}: {used_mib} MiB VRAM is already "
            f"used and GPU busy is {busy}%. Close GPU applications or explicitly pass "
            "--allow-busy-gpu for a non-comparable diagnostic run."
        )


def resolve_checkpoint(config: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        checkpoint = explicit
    else:
        checkpoints = Path(config["run_dir"]) / "checkpoints"
        pointer = json.loads((checkpoints / "latest.json").read_text(encoding="utf-8"))
        checkpoint = checkpoints / pointer["checkpoint_dir"]
    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    if state["sequence_in_block"] != 0:
        raise SystemExit("benchmark checkpoint is not at a safe work-block boundary")
    return checkpoint.resolve()


def run_variant(
    base: dict,
    checkpoint: Path,
    root: Path,
    name: str,
    micro_batch: int,
    accumulation: int,
    device: str,
    updates: int,
) -> Result:
    config = json.loads(json.dumps(base))
    run_dir = root / name
    config_path = root / f"{name}.json"
    config["run_dir"] = str(run_dir)
    config["schedule_batch_multiple"] = base.get(
        "schedule_batch_multiple", base["optimizer"]["micro_batch_size"]
    )
    config["optimizer"]["micro_batch_size"] = micro_batch
    config["optimizer"]["gradient_accumulation"] = accumulation
    config["checkpoint_every_steps"] = 1_000_000_000
    config["validation_every_steps"] = 1_000_000_000
    config["log_every_steps"] = 1
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [
            "./target/release/train_llm",
            "--config",
            str(config_path),
            "--import-checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--max-steps",
            str(updates),
        ],
        check=True,
    )

    events = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [event for event in events if event.get("event") == "train"]
    # Discard the first complete work block: it includes allocator growth and
    # matmul autotuning which do not represent sustained throughput.
    steady = train[4:] if len(train) > 4 else train[1:]
    return Result(
        name=name,
        micro_batch=micro_batch,
        accumulation=accumulation,
        median_tokens_per_second=statistics.median(
            event["tokens_per_second"] for event in steady
        ),
        peak_vram_mib=max(event.get("gpu_peak_vram_mib", 0) for event in train),
        peak_gtt_mib=max(event.get("gpu_peak_gtt_mib", 0) for event in train),
        finite=all(math.isfinite(event["loss"]) for event in train),
    )


def print_results(
    results: list[Result], threshold: int, checkpoint: Path, primary_config: Path
) -> None:
    print("\nvariant  batch  finite  median tok/s  peak VRAM MiB  peak GTT MiB")
    for result in results:
        print(
            f"{result.name:<8} {result.micro_batch}x{result.accumulation:<3} "
            f"{str(result.finite):<7} {result.median_tokens_per_second:>12.0f} "
            f"{result.peak_vram_mib:>14} {result.peak_gtt_mib:>13}"
        )
    eligible = [
        result for result in results if result.finite and result.peak_gtt_mib <= threshold
    ]
    if not eligible:
        print(f"No variant stayed below the {threshold} MiB GTT threshold.")
        return
    selected = max(eligible, key=lambda result: result.median_tokens_per_second)
    print(
        f"Selected physical batch: {selected.micro_batch}x{selected.accumulation} "
        f"({selected.name})."
    )
    if selected.micro_batch == 4:
        print(
            "Resume command:\n  cargo run --release --bin train_llm -- --config "
            + shlex.quote(str(primary_config))
        )
    else:
        print(
            "First fallback continuation:\n"
            "  cargo run --release --bin train_llm -- "
            "--config configs/rx6700-v2-tbptt2-mb2.json "
            "--import-checkpoint "
            + shlex.quote(str(checkpoint))
        )


def main() -> None:
    args = parse_args()
    if args.updates <= 4:
        raise SystemExit("--updates must exceed the four-update warm-up block")
    refuse_while_training()
    refuse_busy_gpu(args.max_idle_vram_mib, args.allow_busy_gpu)
    base = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = resolve_checkpoint(base, args.checkpoint)
    effective_batch = (
        base["optimizer"]["micro_batch_size"]
        * base["optimizer"]["gradient_accumulation"]
    )
    steps_per_block = base["block_sequences"] // effective_batch
    if args.updates % steps_per_block:
        raise SystemExit(
            f"--updates must be divisible by {steps_per_block} to stop on a work-block boundary"
        )

    subprocess.run(["cargo", "build", "--release", "--bin", "train_llm"], check=True)
    with tempfile.TemporaryDirectory(prefix="bdh-cq-v2-memory-benchmark-") as temporary:
        root = Path(temporary)
        results = [
            run_variant(base, checkpoint, root, "mb4", 4, 16, args.device, args.updates)
        ]
        if results[0].peak_gtt_mib > args.gtt_threshold_mib:
            results.append(
                run_variant(base, checkpoint, root, "mb2", 2, 32, args.device, args.updates)
            )
        print_results(results, args.gtt_threshold_mib, checkpoint, args.config)


if __name__ == "__main__":
    main()
