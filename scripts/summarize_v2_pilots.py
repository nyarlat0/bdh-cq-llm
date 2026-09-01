#!/usr/bin/env python3
"""Compare fixed-budget v2 pilot logs without third-party dependencies."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


PILOTS = ("additive", "attnres", "state", "attnres-state", "attnres-state-h1")


def load_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    rows: list[dict] = []
    for name in PILOTS:
        events = load_events(Path(f"runs/rx6700-v2-pilot-tbptt1-{name}/train.jsonl"))
        train = [event for event in events if event.get("event") == "train"]
        validation = [event for event in events if event.get("event") == "validation"]
        finite = bool(train and validation) and all(
            math.isfinite(float(event.get("loss", event.get("selected_loss", math.nan))))
            for event in train + validation
        )
        stateful_validation = [
            event for event in validation if event.get("stateful_loss") is not None
        ]
        best = min(
            (float(event["selected_loss"]) for event in stateful_validation),
            default=None,
        )
        last = stateful_validation[-1] if stateful_validation else None
        recent_speed = [float(event["tokens_per_second"]) for event in train[-20:]]
        rows.append(
            {
                "name": name,
                "finite": finite,
                "best": best,
                "last": None if last is None else float(last["selected_loss"]),
                "fineweb": None
                if last is None
                else last["per_source"]["fineweb2_hq"].get("stateful_bits_per_byte"),
                "ficbook": None
                if last is None
                else last["per_source"]["ficbook"].get("stateful_bits_per_byte"),
                "classic": None
                if last is None
                else last["per_source"]["ru_classic"].get("stateful_bits_per_byte"),
                "speed": statistics.median(recent_speed) if recent_speed else None,
            }
        )

    print(
        "pilot                finite  best-loss  last-loss  FineWeb-BPB  Ficbook-BPB  Classic-BPB  tok/s"
    )
    for row in rows:
        print(
            f"{row['name']:<20} {str(row['finite']):<7} "
            f"{fmt(row['best']):>9}  {fmt(row['last']):>9}  "
            f"{fmt(row['fineweb'], 3):>11}  {fmt(row['ficbook'], 3):>11}  "
            f"{fmt(row['classic'], 3):>11}  {fmt(row['speed'], 0):>5}"
        )

    eligible = [row for row in rows if row["finite"] and row["best"] is not None]
    if eligible:
        winner = min(eligible, key=lambda row: row["best"])
        print(f"\nLowest held-out stateful loss: {winner['name']} ({winner['best']:.5f}).")
        print("Treat differences below ~1% as inconclusive and prefer the faster/simpler variant.")
    else:
        print("\nNo complete stateful pilot is available yet.")


if __name__ == "__main__":
    main()
