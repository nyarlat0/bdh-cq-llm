#!/usr/bin/env python3
"""Summarize the isolated 64/192/384-coordinate RoPE pilot."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


WIDTHS = (64, 192, 384)


def events_for(width: int) -> list[dict]:
    path = Path(f"runs/rx6700-v2-rope-{width}/train.jsonl")
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    print("RoPE/Q  finite  best-loss  last-loss  FineWeb-BPB  Ficbook-BPB  Classic-BPB  tok/s")
    for width in WIDTHS:
        events = events_for(width)
        train = [event for event in events if event.get("event") == "train"]
        validation = [
            event
            for event in events
            if event.get("event") == "validation" and event.get("stateful_loss") is not None
        ]
        finite = bool(train and validation) and all(
            math.isfinite(float(event.get("loss", event.get("selected_loss", math.nan))))
            for event in train + validation
        )
        last = validation[-1] if validation else None
        best = min((float(event["selected_loss"]) for event in validation), default=None)
        speed_values = [float(event["tokens_per_second"]) for event in train[-20:]]
        speed = statistics.median(speed_values) if speed_values else None
        per_source = {} if last is None else last["per_source"]
        print(
            f"{width:>4}/768 {str(finite):<7} {fmt(best):>9}  "
            f"{fmt(None if last is None else float(last['selected_loss'])):>9}  "
            f"{fmt(per_source.get('fineweb2_hq', {}).get('stateful_bits_per_byte'), 3):>11}  "
            f"{fmt(per_source.get('ficbook', {}).get('stateful_bits_per_byte'), 3):>11}  "
            f"{fmt(per_source.get('ru_classic', {}).get('stateful_bits_per_byte'), 3):>11}  "
            f"{fmt(speed, 0):>5}"
        )


if __name__ == "__main__":
    main()
