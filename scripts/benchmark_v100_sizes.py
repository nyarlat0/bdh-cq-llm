#!/usr/bin/env python3
"""Bounded FP32 width/batch sweep; never resume or alter production runs.

Scale D and Q together, preserving Q/D and rotary_dim/Q, not recurrent depth
(the block is shared, so adding depth is not parameter scaling). Each case is
a fresh process with the same effective token budget and CQ active immediately.
"""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from benchmark_v100 import command, read_gpu


def make_config(base, dim, batch, run_dir):
    """Change widths only; retain attention types, head counts and all gates."""
    config = copy.deepcopy(base)
    model = config["model"]
    wide, remainder = divmod(model["dim_qk_heads"] * dim, model["dim"])
    heads = model["heads"]
    if remainder or wide % heads or (wide // heads) % 4:
        raise ValueError("width must preserve Q/D and permit even RoPE=Q/2")
    if dim % max(1, model["attn_residual_heads"]):
        raise ValueError("D must be divisible by MHAR heads")
    model.update(dim=dim, dim_qk_heads=wide, rotary_dim=wide // heads // 2)
    config["run_dir"] = str(run_dir)
    config["schedule_batch_multiple"] = 32
    config["optimizer"].update(micro_batch_size=batch, gradient_accumulation=64 // batch)
    config["memory"].update(stateful_after_tokens=0, memory_read_ramp_tokens=0)
    config.update(log_every_steps=4, validation_every_steps=10**9,
                  checkpoint_every_steps=10**9)
    return config


def parse_result(content, status, warmup_updates, timed_out=False):
    """Never declare a failed/NaN run eligible based on earlier finite lines."""
    samples = re.findall(
        r"step\s+(\d+)\s*\|[^\n]*?loss\s+(\S+)[^\n]*?\|\s+([\d.]+) tok/s", content)
    numeric_failure = bool(re.search(r"non-finite|\b(?:nan|inf(?:inity)?)\b", content, re.I))
    finite = bool(samples) and not numeric_failure and all(
        math.isfinite(float(loss)) for _, loss, _ in samples)
    completed = status == 0 and "requested --max-steps reached" in content and not timed_out
    speeds = [float(speed) for step, _, speed in samples if int(step) > warmup_updates]
    eligible = completed and finite and bool(speeds)
    oom = bool(re.search(r"can't allocate buffer|out of memory|CUDA_ERROR_OUT_OF_MEMORY", content, re.I))
    reason = ("timeout" if timed_out else "allocation_failure" if oom else
              "non_finite" if numeric_failure else "ok" if eligible else "incomplete_or_no_samples")
    counts = re.findall(r"model parameters:\s*(\d+)", content)
    return dict(completed=completed, finite=finite, eligible=eligible, reason=reason,
                parameters=int(counts[-1]) if counts else None,
                median_tok_s=statistics.median(speeds) if eligible else None,
                measured_intervals=len(speeds), exit_code=status)


def peak_vram(path):
    """Sampled process-external GPU usage, not allocator reservations alone."""
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(float(line.split(",")[2].strip()))
        except (ValueError, IndexError):
            pass
    return max(values) if values else None


def report(output, results):
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# V100 FP32 parameter sweep", "",
             "Median excludes warm-up. Failed cases have no eligible speed. VRAM is sampled once/second.", "",
             "| D | Q | RoPE | Parameters | Batch | tok/s | Peak MiB | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for row in results:
        lines.append("| " + " | ".join(str(row.get(key)) for key in
                     ("dim", "q", "rope", "parameters", "batch", "median_tok_s", "peak_vram_mib", "reason")) + " |")
    winners = []
    for dim in sorted({r["dim"] for r in results}):
        candidates = [r for r in results if r["dim"] == dim and r["eligible"]]
        if candidates:
            winners.append(max(candidates, key=lambda r: r["batch"]))
    lines += ["", "## Largest successful batch per width", "",
              "Maximum among tested batches compatible with effective batch=64 and TBPTT=2, not arbitrary integer batches. "
              "NaN/timeouts do not establish a VRAM limit. Compare equal batch rows to isolate width cost.", ""]
    for row in winners:
        ratio = row["median_tok_s"] / winners[0]["median_tok_s"]
        hours = 1e9 / row["median_tok_s"] / 3600
        lines.append(f"- D={row['dim']}, B={row['batch']}: {row['median_tok_s']:.0f} tok/s, "
                     f"{ratio:.2f}x first successful width; 1B tokens ~{hours:.1f} h excluding validation/checkpoints.")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/v100-v2.json"))
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 640, 768, 1024])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="ascending divisors of 32 (effective batch=64, TBPTT=2); stop at failure")
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=900,
                        help="hard per-case limit; timed-out benchmark may not save a checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="generate configs only, no CUDA/build/training")
    args = parser.parse_args()
    base = json.loads(args.config.read_text())
    if (args.updates % 4 or args.warmup_updates % 4 or args.warmup_updates < 4
            or args.updates < args.warmup_updates + 8 or args.timeout_seconds <= 0):
        parser.error("updates/warmup must be multiples of 4; warmup >=4 and at least 8 measured updates")
    if any(b not in (1, 2, 4, 8, 16, 32) for b in args.batches) or any(d <= 0 for d in args.dims):
        parser.error("use positive D and batches dividing 32; accumulation must be divisible by TBPTT=2")
    if args.batches != sorted(args.batches):
        parser.error("batches must increase so failures stop the capacity search")
    if len(set(args.dims)) != len(args.dims) or len(set(args.batches)) != len(args.batches):
        parser.error("duplicate cases are not allowed")
    if base["sequence_length"] != 256 or base["memory"]["chunks_per_detach"] != 2:
        parser.error("expected context=256 and TBPTT=2")
    if base["model"]["rotary_dim"] * 2 * base["model"]["heads"] != base["model"]["dim_qk_heads"]:
        parser.error("base config must already use RoPE=Q/2")
    # Validate widths before creating files, compiling or touching the GPU.
    for dim in args.dims:
        make_config(base, dim, 1, "unused")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or Path("runs") / f"v100-sizes-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for dim in args.dims:
        for batch in args.batches:
            label = f"fp32-d{dim}-b{batch}"
            config = make_config(base, dim, batch, output / label)
            path = output / f"{label}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            cases.append((label, config, path))
    print(f"{len(cases)} cases, {args.updates * 16384:,} tokens each; configs: {output}", flush=True)
    if args.dry_run:
        return
    nvcc = command(["nvcc", "--version"], capture_output=True).stdout
    if not re.search(r"release 12\.9\b", nvcc):
        raise SystemExit("Select CUDA Toolkit 12.9, including NVRTC/libraries.")
    uuid, gpu = read_gpu(args.device)
    env = dict(os.environ, CUDARC_CUDA_VERSION="12090", CUDA_VISIBLE_DEVICES=uuid)
    (output / "hardware.txt").write_text(gpu + "\n" + nvcc)
    # Explicitly exclude experimental FP16, even if it was built previously.
    command(["cargo", "test", "--release", "--locked", "--no-default-features", "--features", "cuda",
             "--test", "precision", "--", "--ignored", "--test-threads=1"], env=env)
    command(["cargo", "build", "--release", "--locked", "--no-default-features", "--features", "cuda",
             "--bin", "train_llm"], env=env)
    target = Path(json.loads(command(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                                    capture_output=True, env=env).stdout)["target_directory"])
    binary = output / "train-fp32"
    shutil.copy2(target / "release/train_llm", binary)
    results = []
    stopped_dims = set()
    for label, config, path in cases:
        if config["model"]["dim"] in stopped_dims:
            continue
        print(f"Starting {label}; log: {output / (label + '.log')}", flush=True)
        telemetry_path = output / f"{label}.gpu.csv"
        started = time.monotonic()
        timed_out = False
        with telemetry_path.open("w") as telemetry:
            monitor = subprocess.Popen(["nvidia-smi", "-i", uuid,
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
                "--format=csv,nounits", "-l", "1"], stdout=telemetry, stderr=subprocess.STDOUT)
            try:
                with (output / f"{label}.log").open("w") as log:
                    try:
                        status = subprocess.run([str(binary), "--config", str(path), "--device", "0",
                            "--max-steps", str(args.updates)], env=env, stdout=log, stderr=subprocess.STDOUT,
                            timeout=args.timeout_seconds).returncode
                    except subprocess.TimeoutExpired:
                        timed_out, status = True, None
            finally:
                monitor.terminate()
                monitor.wait()
        result = parse_result((output / f"{label}.log").read_text(), status, args.warmup_updates, timed_out)
        model = config["model"]
        result.update(dim=model["dim"], q=model["dim_qk_heads"] // model["heads"], rope=model["rotary_dim"],
                      batch=config["optimizer"]["micro_batch_size"],
                      peak_vram_mib=peak_vram(telemetry_path), elapsed_seconds=time.monotonic() - started)
        results.append(result)
        report(output, results)
        print(result, flush=True)
        if not result["eligible"]:
            stopped_dims.add(model["dim"])
            print(f"Stopping D={model['dim']}: {result['reason']}; "
                  "only allocation_failure indicates a measured capacity failure.", flush=True)
    print(f"Summary: {output / 'summary.md'}. No production training started.")


if __name__ == "__main__":
    main()
