#!/usr/bin/env python3
"""Fresh equal-token FP32 learning pilots at previously measured fastest batches.

No production config/checkpoint is changed. Wall times include initialization,
validation and earlier saves, measured independently for each model process.
"""
import argparse
import csv
import fcntl
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from benchmark_v100 import command, read_gpu
from benchmark_v100_sizes import make_config, parse_result


CASES = ((512, 8), (640, 8), (768, 8), (1024, 4))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_step(run_dir):
    pointer = run_dir / "checkpoints/latest.json"
    if not pointer.exists():
        return 0
    latest = json.loads(pointer.read_text())
    name = latest["checkpoint_dir"]
    if Path(name).name != name:
        raise SystemExit("invalid checkpoint pointer")
    state = json.loads((pointer.parent / name / "state.json").read_text())
    step = state["optimizer_step"]
    if step != latest["optimizer_step"]:
        raise SystemExit("checkpoint pointer/state step mismatch")
    return step


def load_events(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return events


def prune_curve(path, dim, committed_step):
    """Keep only committed trajectory points; retain original CSV as evidence."""
    if not path.exists():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    kept = [r for r in rows if int(r["dim"]) != dim or int(r["step"]) <= committed_step]
    if kept != rows:
        shutil.copy2(path, path.with_name(f"validation-before-resume-{time.time_ns()}.csv"))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)


def prune_events(path, committed_step):
    if not path.exists():
        return
    events = [e for e in load_events(path) if e.get("step", 0) <= committed_step]
    text = "".join(json.dumps(e) + "\n" for e in events)
    if text != path.read_text():
        shutil.copy2(path, path.with_name(f"train-before-resume-{time.time_ns()}.jsonl"))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text)
        temporary.replace(path)


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
             "| D | Batch | Params | Completed | Eligible | Best loss | Last loss | tok/s | Wall hours | Approx time |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key)) for key in
                     ("dim", "batch", "parameters", "completed", "eligible", "best_loss",
                      "last_loss", "median_tok_s", "wall_hours", "wall_time_approximate")) + " |")
    lines += ["", "Use validation.csv for loss versus tokens and wall_seconds; "
              "per-source BPB and memoryless/stateful metrics remain in each train.jsonl. "
              "Wall time includes initialization, validation and preceding checkpoints. "
              "Compare first threshold crossings in CSV, not just final loss, for time-to-quality."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/v100-v2.json"))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--validation-every", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="existing pilot directory; keep its frozen configs/binary")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume and (args.output or args.dry_run):
        parser.error("--resume cannot be combined with --output or --dry-run")
    manifest = None
    if args.resume:
        output = args.resume.resolve(strict=True)
        manifest_path = output / "pilot.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if args.updates is not None and args.updates != manifest["updates"]:
                parser.error("--updates differs from the saved total budget")
            args.updates = manifest["updates"]
        elif args.updates is None:
            parser.error("old pilot has no saved budget; specify its ORIGINAL --updates (default was 3072)")
    args.updates = args.updates if args.updates is not None else 3072
    if (args.validation_every < 16 or args.validation_every % 16
            or args.updates < args.validation_every or args.updates % args.validation_every):
        parser.error("validation interval must be a multiple of 16; updates a positive multiple of interval")
    base = json.loads((output / "fp32-d512-b8.json" if args.resume else args.config).read_text())
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
    if not args.resume:
        output = (args.output or Path("runs") / f"v100-size-pilots-{stamp}").resolve()
        output.mkdir(parents=True, exist_ok=False)
    # Held throughout the sequence; a second harness cannot launch duplicate work.
    lock = (output / ".pilot.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another pilot harness owns this directory")
    cases = []
    for dim, batch in CASES:
        label = f"fp32-d{dim}-b{batch}"
        path = output / f"{label}.json"
        if args.resume:
            config = json.loads(path.read_text())
            if Path(config["run_dir"]).resolve() != output / label:
                parser.error(f"saved run_dir in {path} does not match resume location")
            if args.updates % config["validation_every_steps"]:
                parser.error("total budget must end on a saved-config validation interval")
            if manifest and config != manifest["configs"][label]:
                parser.error(f"saved config changed: {path}")
        else:
            config = pilot_config(base, dim, batch, output / label, args.validation_every)
            path.write_text(json.dumps(config, indent=2) + "\n")
        cases.append((dim, batch, label, path))
    print(f"Four pilots, {args.updates * 16384:,} tokens each; {output}", flush=True)
    if args.dry_run:
        atomic_json(output / "pilot.json", dict(updates=args.updates,
                    configs={label: json.loads(path.read_text()) for _, _, label, path in cases}))
        lock.close()
        return
    nvcc = command(["nvcc", "--version"], capture_output=True).stdout
    if not re.search(r"release 12\.9\b", nvcc):
        raise SystemExit("Select CUDA Toolkit 12.9, including NVRTC/libraries.")
    uuid, gpu = read_gpu(args.device)
    if manifest is None:
        manifest = dict(updates=args.updates,
                        configs={label: json.loads(path.read_text()) for _, _, label, path in cases})
        atomic_json(output / "pilot.json", manifest)
    env = dict(os.environ, CUDARC_CUDA_VERSION="12090", CUDA_VISIBLE_DEVICES=uuid)
    (output / f"hardware-{stamp}.txt").write_text(gpu + "\n" + nvcc)
    binary = output / "train-fp32"
    if not args.resume:
        command(["cargo", "build", "--release", "--locked", "--no-default-features", "--features", "cuda",
                 "--bin", "train_llm"], env=env)
        target = Path(json.loads(command(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                                        capture_output=True, env=env).stdout)["target_directory"])
        shutil.copy2(target / "release/train_llm", binary)
    elif not binary.is_file():
        parser.error("saved train-fp32 binary is missing; refusing to change execution code silently")
    if args.resume:
        # Archive acknowledged STOP requests only after confirming an idle GPU.
        for stop in [output / "STOP"] + [output / label / "STOP" for _, _, label, _ in cases]:
            if stop.is_file():
                stop.rename(stop.with_name(f"STOP.acknowledged-{stamp}"))
    rows_path = output / "results.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    for dim, batch, label, path in cases:
        step = checkpoint_step(output / label)
        if step > args.updates:
            parser.error(f"{label} checkpoint exceeds total budget")
        prune_curve(output / "validation.csv", dim, step)
    curve_path = output / "validation.csv"
    new_curve = not curve_path.exists() or curve_path.stat().st_size == 0
    with curve_path.open("a", newline="") as curve:
        writer = csv.writer(curve)
        if new_curve:
            writer.writerow(["dim", "batch", "step", "tokens", "wall_seconds", "memoryless_loss", "stateful_loss"])
        curve.flush()
        for dim, batch, label, path in cases:
            committed = checkpoint_step(output / label)
            if committed == args.updates:
                events = load_events(output / label / "train.jsonl")
                if not any(e.get("event") == "validation" and e["step"] == committed for e in events):
                    raise SystemExit(f"{label}: final checkpoint exists but final validation is missing")
                print(f"Skipping completed {label}, step {committed}", flush=True)
                if not any(r["dim"] == dim and r.get("eligible") for r in rows):
                    validation = [e for e in events if e.get("event") == "validation" and e["step"] <= committed]
                    content = (output / f"{label}.log").read_text()
                    row = parse_result(content, 0, 32)
                    final = validation[-1]
                    row.update(dim=dim, batch=batch, completed=True,
                               eligible=all(math.isfinite(e["selected_loss"]) for e in validation),
                               best_loss=min(e["selected_loss"] for e in validation), last_loss=final["selected_loss"],
                               wall_hours=max((e.get("elapsed_seconds", 0) for e in events), default=0)/3600,
                               wall_time_approximate=True)
                    rows = [r for r in rows if r["dim"] != dim] + [row]
                    write_summary(output, rows)
                continue
            if (output / "STOP").exists():
                raise SystemExit("pilot sequence STOP requested")
            clock_path = output / f"{label}.time.json"
            if clock_path.exists():
                clock = json.loads(clock_path.read_text())
            else:
                old_events = load_events(output / label / "train.jsonl")
                elapsed = max((e.get("elapsed_seconds", 0) for e in old_events), default=0)
                with curve_path.open(newline="") as old_curve:
                    elapsed = max([elapsed] + [float(r["wall_seconds"]) for r in csv.DictReader(old_curve)
                                               if int(r["dim"]) == dim])
                # Old harness lacked a durable clock. Use a known lower bound,
                # never invent the unobserved time between its last log and exit.
                clock = dict(seconds=elapsed, approximate=bool(old_events))
            offset = clock["seconds"]
            prune_events(output / label / "train.jsonl", committed)
            if clock.get("active"):
                clock["approximate"] = True
            session_log = output / f"{label}-attempt-{time.time_ns()}.log"
            print(f"Starting {label}; log: {output / (label + '.log')}", flush=True)
            started = time.monotonic()
            clock["active"] = True
            atomic_json(clock_path, clock)
            # Inherit stdout's lines into both the terminal and a retained log.
            # A global STOP file stops the whole sequence, not only this case.
            with session_log.open("w") as attempt, (output / f"{label}.log").open("a") as log:
                process = subprocess.Popen([str(binary), "--config", str(path), "--device", "0",
                    "--max-steps", str(args.updates - committed)], env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        attempt.write(line)
                        attempt.flush()
                        clock["seconds"] = offset + time.monotonic() - started
                        atomic_json(clock_path, clock)
                        if (output / "STOP").exists():
                            (output / label / "STOP").touch()
                        match = re.search(r"validation step (\d+): memoryless ([\d.eE+-]+), stateful Some\(([\d.eE+-]+)\)", line)
                        if match:
                            step, memoryless, stateful = match.groups()
                            writer.writerow([dim, batch, step, int(step) * 16384,
                                             clock["seconds"], memoryless, stateful])
                            curve.flush()
                    status = process.wait()
                except KeyboardInterrupt:
                    # Foreground Ctrl-C also reaches the trainer's graceful
                    # handler. Do not start another pilot after user cancellation.
                    print(f"Pilot sequence cancelled; trainer PID {process.pid}; "
                          f"wait for its safe checkpoint in {output / label}.", flush=True)
                    raise SystemExit(130)
            clock.update(seconds=offset + time.monotonic() - started, active=False)
            atomic_json(clock_path, clock)
            content = session_log.read_text()
            row = parse_result(content, status, committed + 32)
            events_path = output / label / "train.jsonl"
            events = load_events(events_path)
            validation = [e for e in events if e.get("event") == "validation"]
            row["eligible"] = bool(row["completed"] and row["finite"] and validation
                                   and validation[-1]["step"] == args.updates
                                   and all(math.isfinite(e["selected_loss"]) for e in validation)
                                   and checkpoint_step(output / label) == args.updates)
            row.update(dim=dim, batch=batch, wall_hours=clock["seconds"] / 3600,
                       wall_time_approximate=clock["approximate"],
                       best_loss=min((e["selected_loss"] for e in validation), default=None),
                       last_loss=validation[-1]["selected_loss"] if validation else None)
            rows = [r for r in rows if r["dim"] != dim] + [row]
            write_summary(output, rows)
            # No automatic smaller-batch retry: that would silently change the
            # agreed experiment. STOP/non-finite/OOM all stop the sequence.
            if not row["eligible"] or (output / "STOP").exists():
                raise SystemExit(f"Sequence stopped; inspect {output / 'summary.md'} and {label}.log")
    print(f"Done: {output / 'summary.md'}. No production training started.")
    lock.close()


if __name__ == "__main__":
    main()
