#!/usr/bin/env python3
"""User-run V100 preflight, CUDA smoke tests, and bounded fresh-run batch sweep.

No production run is resumed or modified. Results, configs, telemetry and final
checkpoints are retained under a new directory. The CQ threshold is overridden
ONLY in these disposable benchmark configs to measure the stateful hot path.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def read_gpu(index):
    fields = "uuid,name,memory.total,compute_cap"
    row = command(["nvidia-smi", "-i", str(index), f"--query-gpu={fields}",
                   "--format=csv,noheader,nounits"], capture_output=True).stdout.strip()
    uuid, name, memory, capability = [item.strip() for item in row.split(",")]
    if "V100" not in name or capability != "7.0" or int(memory) < 30000:
        raise SystemExit(f"Expected one V100 32GB, got: {row}")
    running = command(["nvidia-smi", "-i", uuid,
                       "--query-compute-apps=pid", "--format=csv,noheader"],
                      capture_output=True).stdout.strip()
    if running:
        raise SystemExit(f"GPU already has compute processes: {running}")
    return uuid, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="physical nvidia-smi index")
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--config", type=Path, default=Path("configs/v100-v2.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.updates < 8:
        parser.error("use at least 8 updates; first four logged updates are excluded")
    if not Path("Cargo.toml").is_file():
        parser.error("run from the project root")
    nvcc = command(["nvcc", "--version"], capture_output=True).stdout
    if not re.search(r"release 12\.9\b", nvcc):
        raise SystemExit("Select CUDA Toolkit 12.9 (including NVRTC) in PATH/library path, not CUDA 13.")
    uuid, gpu = read_gpu(args.device)
    print(f"Selected GPU: {gpu}", flush=True)
    env = dict(os.environ, CUDARC_CUDA_VERSION="12090", CUDA_VISIBLE_DEVICES=uuid)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or Path("runs") / f"v100-benchmark-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "hardware.txt").write_text(gpu + "\n" + nvcc)
    base = json.loads(args.config.read_text())
    if base["sequence_length"] != 256 or base["memory"]["chunks_per_detach"] != 2:
        parser.error("this sweep expects sequence_length=256 and TBPTT=2")
    if base["optimizer"]["micro_batch_size"] * base["optimizer"]["gradient_accumulation"] != 64:
        parser.error("this sweep preserves 64 sequences / 16384 tokens per update")
    results = []
    target = Path(json.loads(command(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                                    capture_output=True, env=env).stdout)["target_directory"])
    for mode, feature in [("fp32", "cuda"), ("fp16-projections", "cuda-fp16")]:
        # --test precision keeps GPU tests explicit. No CUDA test runs in the
        # ordinary test suite on the developer's AMD machine.
        command(["cargo", "test", "--release", "--locked", "--features", feature,
                 "--test", "precision", "--", "--ignored", "--test-threads=1", "--nocapture"], env=env)
        if args.smoke_only:
            continue
        command(["cargo", "build", "--release", "--locked", "--features", feature,
                 "--bin", "train_llm"], env=env)
        binary = output / f"train-{mode}"
        shutil.copy2(target / "release/train_llm", binary)
        for batch in [4, 8, 16]:
            label = f"{mode}-b{batch}"
            print(f"Starting {label}, {args.updates} updates; log: {output / (label + '.log')}", flush=True)
            config = json.loads(json.dumps(base))
            config["run_dir"] = str(output / label)
            config["schedule_batch_multiple"] = 16
            config["optimizer"]["micro_batch_size"] = batch
            config["optimizer"]["gradient_accumulation"] = 64 // batch
            config["memory"]["stateful_after_tokens"] = 0
            config["memory"]["memory_read_ramp_tokens"] = 0
            # Avoid forcing CQ/gate diagnostics and their host readbacks on
            # every update. Each throughput sample covers four updates.
            config["log_every_steps"] = 4
            config["validation_every_steps"] = 1000000000
            config["checkpoint_every_steps"] = 1000000000
            config_path = output / f"{label}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            with (output / f"{label}.gpu.csv").open("w") as telemetry:
                monitor = subprocess.Popen(["nvidia-smi", "-i", uuid,
                    "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
                    "--format=csv", "-l", "1"], stdout=telemetry, stderr=subprocess.STDOUT)
                try:
                    with (output / f"{label}.log").open("w") as log:
                        status = subprocess.run([str(binary), "--config", str(config_path),
                            "--device", "0", "--max-steps", str(args.updates)],
                            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                finally:
                    monitor.terminate()
                    monitor.wait()
            # Parse the actual console schema; don't confuse a successful
            # launch with a completed run or count checkpoint serialization.
            content = (output / f"{label}.log").read_text()
            samples = re.findall(r"loss ([\d.eE+\-]+).*?\| ([\d.]+) tok/s", content)
            finite = bool(samples) and all(math.isfinite(float(loss)) for loss, _ in samples)
            completed = status == 0 and "requested --max-steps reached" in content
            measured = [float(speed) for _, speed in samples[1:]]
            result = dict(mode=mode, batch=batch, accumulation=64//batch,
                          completed=completed, finite=finite,
                          median_tok_s=statistics.median(measured) if measured else None)
            results.append(result)
            print(result, flush=True)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if status != 0:
                print(f"Failed (possibly OOM): inspect {label}.log; not eligible for selection.")
                # Larger microbatches won't fix an OOM or a kernel failure.
                break
    print(f"Reports retained in {output}. No production training started.")


if __name__ == "__main__":
    main()
