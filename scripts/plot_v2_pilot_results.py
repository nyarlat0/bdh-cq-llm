#!/usr/bin/env python3
"""Render the completed architecture and RoPE pilot curves as plain SVG.

The script intentionally uses only Python's standard library.  Its inputs are
the immutable JSONL logs produced by the fixed-budget launchers, and its two
outputs are documentation assets under ``docs/assets``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
ARCHITECTURES = {
    "additive": "#64748b",
    "attnres": "#2563eb",
    "state": "#16a34a",
    "attnres-state": "#dc2626",
    "attnres-state-h1": "#9333ea",
}
ROPE_WIDTHS = {
    64: "#64748b",
    192: "#2563eb",
    384: "#dc2626",
}


def stateful_validation(path: Path) -> list[tuple[float, float]]:
    """Return ``(millions of training tokens, stateful loss)`` points."""

    if not path.is_file():
        raise SystemExit(f"missing pilot log: {path.relative_to(ROOT)}")
    points = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") != "validation" or event.get("stateful_loss") is None:
                continue
            points.append((float(event["tokens_seen"]) / 1_000_000.0, float(event["stateful_loss"])))
    if not points:
        raise SystemExit(f"no stateful validation events in {path.relative_to(ROOT)}")
    return points


def svg_chart(
    series: dict[str, tuple[str, list[tuple[float, float]]]],
    title: str,
    subtitle: str,
    y_label: str,
    output: Path,
) -> None:
    """Draw a deterministic line chart suitable for Markdown documentation."""

    width, height = 960, 560
    left, right, top, bottom = 88, 28, 86, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [point for _, values in series.values() for point in values]
    x_values = sorted({point[0] for point in points})
    x_min, x_max = min(x_values), max(x_values)
    raw_y_min = min(point[1] for point in points)
    raw_y_max = max(point[1] for point in points)
    y_padding = max((raw_y_max - raw_y_min) * 0.08, 0.001)
    y_min, y_max = raw_y_min - y_padding, raw_y_max + y_padding

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#526071}.tick{font-size:12px;fill:#526071}.legend{font-size:12px}.axis{stroke:#7b8794;stroke-width:1}.grid{stroke:#dce3ea;stroke-width:1}.line{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}.point{stroke:#fff;stroke-width:1.5}</style>',
        f'<text class="title" x="{left}" y="34">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="57">{html.escape(subtitle)}</text>',
    ]

    y_ticks = 6
    for index in range(y_ticks):
        value = y_min + index * (y_max - y_min) / (y_ticks - 1)
        py = y(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}"/>')
        lines.append(f'<text class="tick" x="{left-12}" y="{py+4:.2f}" text-anchor="end">{value:.3f}</text>')

    for value in x_values:
        px = x(value)
        lines.append(f'<line class="grid" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}"/>')
        lines.append(f'<text class="tick" x="{px:.2f}" y="{height-bottom+24}" text-anchor="middle">{value:.3g}</text>')

    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<text class="tick" x="{left + plot_width/2:.2f}" y="{height-24}" text-anchor="middle">training tokens, millions</text>',
            f'<text class="tick" transform="translate(22 {top + plot_height/2:.2f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
        ]
    )

    for name, (color, values) in series.items():
        coordinates = " ".join(f"{x(px):.2f},{y(py):.2f}" for px, py in values)
        dash = ' stroke-dasharray="8 5"' if name.endswith("h1") else ""
        lines.append(f'<polyline class="line" stroke="{color}" points="{coordinates}"{dash}/>')
        for px, py in values:
            lines.append(f'<circle class="point" cx="{x(px):.2f}" cy="{y(py):.2f}" r="3.5" fill="{color}"/>')

    legend_x = left
    legend_y = 78
    for name, (color, _) in series.items():
        label_width = 34 + len(name) * 7
        if legend_x + label_width > width - right:
            legend_x = left
            legend_y += 20
        dash = ' stroke-dasharray="6 4"' if name.endswith("h1") else ""
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+20}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash}/>')
        lines.append(f'<text class="legend" x="{legend_x+26}" y="{legend_y+4}">{html.escape(name)}</text>')
        legend_x += label_width

    lines.append(f'<text class="tick" x="{width-right}" y="{height-10}" text-anchor="end">generated from fixed-budget train.jsonl logs</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    architecture = {
        name: (
            color,
            stateful_validation(ROOT / "runs" / f"rx6700-v2-delta-pilot-{name}" / "train.jsonl"),
        )
        for name, color in ARCHITECTURES.items()
    }
    svg_chart(
        architecture,
        "Architecture-v2 pilot: stateful validation loss",
        "Same seed, data order, 1,220 updates and 5M-token CQ activation; lower is better",
        "stateful cross-entropy loss",
        ASSETS / "v2-architecture-validation-loss.svg",
    )

    rope_absolute = {
        width: stateful_validation(ROOT / "runs" / f"rx6700-v2-rope-{width}" / "train.jsonl")
        for width in ROPE_WIDTHS
    }
    baseline = dict(rope_absolute[64])
    rope_delta = {
        f"RoPE {width}": (
            ROPE_WIDTHS[width],
            [(tokens, loss - baseline[tokens]) for tokens, loss in values],
        )
        for width, values in rope_absolute.items()
    }
    svg_chart(
        rope_delta,
        "RoPE-width pilot: validation-loss delta from RoPE 64",
        "Negative values beat the 64/768 baseline; all other settings are identical",
        "loss difference versus RoPE 64",
        ASSETS / "v2-rope-validation-loss-delta.svg",
    )


if __name__ == "__main__":
    main()
