#!/usr/bin/env python3
"""Plot a snapshot of train.jsonl or a trainer console log as standalone SVG.

No dependencies. Partial final JSON lines are ignored; interior malformed JSON
is an error. Restarted trajectories discard points after the resumed cursor.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys

from plot_v2_pilot_results import svg_chart


def read_loss(path):
    series = {key: {} for key in ("train", "memoryless", "stateful")}
    last_train = -1
    lines = path.read_text(errors="replace").splitlines()
    json_mode = next((line.lstrip().startswith("{") for line in lines if line.strip()), False)
    token_by_step = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        event = None
        if json_mode:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    print("Ignoring incomplete final JSON line", file=sys.stderr)
                    continue
                raise ValueError(f"Malformed JSON at line {index+1}")
        else:
            resumed = re.search(r"resumed .*? at step (\d+)", line)
            if resumed:
                cursor = int(resumed[1])
                for points in series.values():
                    for step in list(points):
                        if step > cursor:
                            del points[step]
                last_train = cursor
            train = re.search(r"step\s+(\d+)\s*\|.*?tokens\s+(\d+).*?loss\s+(\S+)", line)
            validation = re.search(r"validation step (\d+): memoryless (\S+), stateful (?:Some\(([^)]+)\)|None)", line)
            if train:
                event = dict(event="train", step=int(train[1]), tokens_seen=int(train[2]), loss=float(train[3]))
            elif validation:
                event = dict(event="validation", step=int(validation[1]),
                             memoryless_loss=float(validation[2]),
                             stateful_loss=float(validation[3]) if validation[3] else None)
        if not event or event.get("event") not in ("train", "validation"):
            continue
        step = int(event["step"])
        if event["event"] == "train":
            if step <= last_train:
                for points in series.values():
                    for old in list(points):
                        if old >= step:
                            del points[old]
            last_train = step
        tokens = event.get("tokens_seen", token_by_step.get(step))
        if tokens is not None:
            token_by_step[step] = tokens
        for key, field in (("train", "loss"), ("memoryless", "memoryless_loss"), ("stateful", "stateful_loss")):
            value = event.get(field)
            if value is not None:
                if math.isfinite(float(value)):
                    series[key][step] = (tokens, float(value))
                else:
                    print(f"WARNING: non-finite {key} loss at step {step}; omitted", file=sys.stderr)
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="train.jsonl, console .log, or run directory")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--x", choices=["steps", "tokens"], default="steps")
    parser.add_argument("--smooth", type=int, default=1, help="trailing training-loss mean in logged points")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")
    path = args.log / "train.jsonl" if args.log.is_dir() else args.log
    raw = read_loss(path)
    curves = {}
    for key, color in (("train", "#94a3b8"), ("memoryless", "#2563eb"), ("stateful", "#dc2626")):
        points, window = [], []
        for step, (tokens, value) in sorted(raw[key].items()):
            if args.x == "tokens" and tokens is None:
                parser.error("token counts missing for validation; use --x steps or train.jsonl")
            window.append(value)
            y = sum(window[-args.smooth:]) / min(len(window), args.smooth) if key == "train" else value
            points.append((step if args.x == "steps" else tokens / 1e6, y))
        if points:
            curves[key] = (color, points)
    if not curves:
        parser.error("no finite loss observations yet")
    output = args.output or path.with_suffix(".loss.svg")
    if output.resolve() == path.resolve():
        parser.error("output must not overwrite the input log")
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_chart(curves, "Training and validation loss", f"{path.name}; training mean window={args.smooth}; snapshot, not live",
              "cross-entropy, nats/token", output, "optimizer steps" if args.x == "steps" else "training tokens, millions")
    print(output)


if __name__ == "__main__":
    main()
