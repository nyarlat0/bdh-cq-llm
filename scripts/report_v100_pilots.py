#!/usr/bin/env python3
"""Reproduce the V100 size-pilot report and SVG curves from copied raw logs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from plot_v2_pilot_results import svg_chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="?", default=Path("runs/v100-size-pilots-20260908-131245-087810"))
    args = parser.parse_args()
    output = Path("docs")
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.runs / "results.json").read_text())
    with (args.runs / "validation.csv").open() as stream:
        timings = {(int(r["dim"]), int(r["step"])): float(r["wall_seconds"]) for r in csv.DictReader(stream)}
    tokens, times, deltas = {}, {}, {}
    table, source_table, thresholds = [], [], []
    baseline = None
    for row, color in zip(sorted(rows, key=lambda r: r["dim"]), ("#2563eb", "#16a34a", "#9333ea", "#dc2626")):
        dim, batch = row["dim"], row["batch"]
        run = args.runs / f"fp32-d{dim}-b{batch}"
        events = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines() if line.strip()]
        train = [e for e in events if e["event"] == "train"]
        validation = [e for e in events if e["event"] == "validation"]
        assert train[-1]["step"] == 3072
        assert json.loads((run / "checkpoints/latest.json").read_text())["optimizer_step"] == 3072
        assert [e["step"] for e in validation] == list(range(256, 3072, 256))
        assert all(math.isfinite(e["loss"]) for e in train)
        assert all(math.isfinite(e["stateful_loss"]) for e in validation)
        label = f"{row['parameters']/1e6:.1f}M"
        tokens[label] = (color, [(e["tokens_seen"]/1e6, e["stateful_loss"]) for e in validation])
        times[label + (" ~time" if row["wall_time_approximate"] else "")] = (
            color, [(timings[dim, e["step"]]/3600, e["stateful_loss"]) for e in validation])
        if baseline is None:
            baseline = {e["step"]: e["stateful_loss"] for e in validation}
        deltas[label] = (color, [(e["tokens_seen"]/1e6, e["stateful_loss"]-baseline[e["step"]]) for e in validation])
        last = validation[-1]
        speed = statistics.median(e["tokens_per_second"] for e in train[-20:])
        approx = "~" if row["wall_time_approximate"] else ""
        table.append(f"| {label} | {dim} | {batch} | {last['stateful_loss']:.5f} | {last['memoryless_loss']:.5f} | {speed:.0f} | {approx}{row['wall_hours']:.2f} |")
        bpb = [last["per_source"][s]["stateful_bits_per_byte"] for s in ("fineweb2_hq", "ficbook", "ru_classic")]
        source_table.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in bpb) + " |")
        first = next(e for e in validation if e["stateful_loss"] <= 7)
        thresholds.append(f"| {label} | {first['tokens_seen']/1e6:.2f} | {approx}{timings[dim,first['step']]/3600:.2f} |")
    svg_chart(tokens, "V100 size pilots: loss versus tokens", "FP32; same recipe and token budget; RoPE = Q/2", "stateful validation loss", assets / "v100-size-loss-tokens.svg")
    svg_chart(times, "V100 size pilots: loss versus elapsed time", "~time: approximate after interruptions; downtime excluded; setup/evaluation included", "stateful validation loss", assets / "v100-size-loss-time.svg", "active elapsed hours (some approximate)")
    svg_chart(deltas, "V100 size pilots: loss difference versus 22M", "Matched validation steps; negative values beat the 22M baseline", "stateful loss difference", assets / "v100-size-loss-delta.svg")
    report = """# V100 model-size pilot results

## Protocol and verification

These experiments evaluate our experimental v2 extensions, not published
BDH-CQ results. All four models completed 3,072 updates, or **50,331,648 tokens**.
Raw train.jsonl files and final checkpoint pointers were checked: step 3072,
192 training records and 11 validations per model, with finite recorded losses.
The last validation is at **step 2816, 46,137,344 tokens**, not the final
checkpoint: the trainer skips validation when it reaches --max-steps.

FP32 CUDA on V100; seed=42, vocabulary=24576, D=512/640/768/1024,
H=8, Q=1.5D, RoPE=Q/2, shared depth=8, MHAR=8. Delta wide-state,
per-neuron gates/retention and normalization are unchanged. TBPTT=2×256.
CQ reads/writes are fully enabled from the start; effective batch=64, physical
batch=8/8/8/4. Every model uses the same LR recipe: 10M-token warm-up,
max=3e-4 and the original full decay horizon. These are early training prefixes,
without size-specific LR tuning. Validation uses the same 384 single-lane
chunks (128 per source). Physical batch changes document grouping across lanes.

## Results

The table reports the last validation loss, also the best for every model.
Throughput is recomputed consistently as the median of the last 20 training
records, rather than different portions of attempt logs. Hours are the saved
total pilot times including overhead. `~` denotes approximate reconstruction
after interruptions; exact wall-time comparisons are unavailable for 22M/63M.

| Parameters | D | Batch | CQ loss | Memoryless loss | tok/s | Hours |
|---|---|---|---|---|---|---|
""" + "\n".join(table) + """

![Loss versus tokens](assets/v100-size-loss-tokens.svg)

![Loss difference versus 22M](assets/v100-size-loss-delta.svg)

| Parameters | FineWeb BPB | Ficbook BPB | Classic BPB |
|---|---|---|---|
""" + "\n".join(source_table) + """

## Time to a quality threshold

![Loss versus elapsed time](assets/v100-size-loss-time.svg)

First **observed** stateful validation loss ≤ 7: the exact crossing between
evaluations is unknown. No interpolation is used.

| Parameters | Tokens, millions | Active hours |
|---|---|---|
""" + "\n".join(thresholds) + """

## Conclusions and limitations

- At equal token counts, the larger model is better on all three sources.
  Relative to 22M, loss falls by 0.10975 / 0.19513 / 0.35264 nats for
  30.5/40.1/63M: approximately 10.4% / 17.7% / 29.7% lower perplexity.
- Larger models need fewer tokens to reach loss ≤ 7, but reach that threshold
  later on the observed time axis. Scaling parameters does not automatically
  improve time to quality on one V100.
- 63M has the best loss at a fixed token budget and the most expensive run.
  22M is useful for quick iterations; 30.5M is an intermediate compromise.
  These pilots do not establish a universal winner for final quality.
- Enabling CQ at evaluation reduces loss by 0.083/0.091/0.101/0.120 nats.
  This compares read-on/read-off in the same model, not separately trained
  memoryless models, and does not prove reliable long-context retrieval.
- One seed, a short budget and a shared LR cannot predict results at 1B tokens,
  compare optimally tuned sizes, or guarantee generation quality. Historical
  RX architecture/RoPE pilots used different TBPTT and CQ settings; their
  absolute losses are not a controlled measure of architectural regression.

## Recommendation for full training

For a quality-first run with the planned approximately 1.05B-token budget,
**choose 63M (D=1024, Q=1536, RoPE=768), FP32, batch=4, accumulation=16**,
provided roughly a week of V100 compute is acceptable. This is a practical
choice based on the best measured equal-token validation results, not proof
that the ranking will persist to the end of pretraining.

At 1,738 tok/s, 1.05B tokens take approximately 168 hours (7 days) of training
alone, plus validation/checkpoints and other overhead. This extrapolates the
observed rate; later data phases may have different throughput. The capacity
sweep measured about 23.9 GiB peak usage for 63M/B4, versus 30.6 GiB for
40M/B8. These are sampled benchmark peaks, not guaranteed long-run bounds.

If turnaround matters more, choose 40M: about 100 hours of training alone at
2,907 tok/s, but B8 leaves little VRAM headroom. No production config is changed
by this recommendation. The faster initial threshold crossing of 22M should
not be mistaken for evidence of better final quality at a fixed token budget.

## Reproducing the report and plotting a single run

```bash
python3 scripts/report_v100_pilots.py
python3 scripts/plot_loss.py runs/RUN_NAME/train.jsonl --x tokens --smooth 10 -o /tmp/loss.svg
python3 scripts/plot_loss.py /path/to/console.log -o /tmp/console-loss.svg
```

Plots are SVG and require only the Python standard library. plot_loss also
accepts a run directory and draws train/memoryless/stateful curves without
requiring completed training. It takes a snapshot; rerunning updates the SVG.
A truncated final JSON line is skipped with a warning; corruption in the
middle is an error. Only training loss is smoothed, over logged records rather
than optimizer steps. Use the steps axis for console logs missing token counts
at validation points. Backward step jumps indicate restarts: later old points
are discarded. Retain original logs; the plot cannot recover missing data.
"""
    (output / "v100-size-pilot-results.md").write_text(report)
    print(output / "v100-size-pilot-results.md")


if __name__ == "__main__":
    main()
