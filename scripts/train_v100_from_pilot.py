#!/usr/bin/env python3
"""Import the completed 22M pilot once, then resume the separate full run.

Run from the project root. Source files are never moved, edited or deleted.
The trainer restores model AND AdamW AND data cursor; --max-steps is omitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


CONFIG = Path("configs/v100-22m-from-pilot.json")
DEFAULT_PILOT = Path("runs/v100-size-pilots-20260908-131245-087810/fp32-d512-b8")
IMPORT_FILES = ("config.json", "checkpoints/step-000000003072/state.json",
                "checkpoints/step-000000003072/model.bin", "checkpoints/step-000000003072/optimizer.bin")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_command(pilot, device, dry_run=False):
    """Snapshot exactly the import files, atomically publishing a verified copy.

    A failed/interrupted copy is left in a uniquely named staging directory for
    inspection; it is never used as a checkpoint. No large datasets are copied.
    """
    run = Path(json.loads(CONFIG.read_text())["run_dir"])
    snapshot = run / "pilot-source"
    if (run / "checkpoints/latest.json").exists():
        return launch_command(pilot, device)
    if snapshot.exists():
        manifest = json.loads((snapshot / "copy-sha256.json").read_text())
        if set(manifest) != set(IMPORT_FILES):
            raise ValueError("copied pilot manifest has unexpected files")
        for name, expected in manifest.items():
            if digest(snapshot / name) != expected:
                raise ValueError(f"copied pilot is corrupted: {name}")
        return launch_command(snapshot, device)
    command = launch_command(pilot, device)  # Validate before creating anything.
    if dry_run:
        print(f"Would copy four pilot files from {pilot} to {snapshot}; datasets stay shared.")
        return command
    run.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pilot-source-copy-", dir=run))
    manifest = {}
    for name in IMPORT_FILES:
        source, destination = pilot / name, staging / name
        expected = digest(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if digest(destination) != expected:
            raise ValueError(f"copy verification failed: {name}; staging retained at {staging}")
        manifest[name] = expected
    (staging / "copy-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    launch_command(staging, device)  # Recheck the copied config/state relationship.
    staging.rename(snapshot)
    print(f"Copied and SHA-256 verified four pilot files in {snapshot}", flush=True)
    return launch_command(snapshot, device)


def launch_command(pilot, device):
    config = json.loads(CONFIG.read_text())
    run = Path(config["run_dir"])
    if (run / "STOP").exists():
        raise ValueError(f"{run / 'STOP'} exists; remove it explicitly before continuing")
    args = ["cargo", "run", "--release", "--locked", "--no-default-features", "--features", "cuda",
            "--bin", "train_llm", "--", "--config", str(CONFIG), "--device", str(device)]
    if (run / "checkpoints/latest.json").exists():
        return args  # The trainer validates the destination's frozen hashes.
    checkpoint = pilot / "checkpoints/step-000000003072"
    for path in [pilot / "config.json", checkpoint / "state.json",
                 checkpoint / "model.bin", checkpoint / "optimizer.bin"]:
        if not path.is_file():
            raise ValueError(f"missing import file: {path}")
    state = json.loads((checkpoint / "state.json").read_text())
    previous_bytes = (pilot / "config.json").read_bytes()
    previous = json.loads(previous_bytes)
    if hashlib.sha256(previous_bytes).hexdigest() != state["config_sha256"]:
        raise ValueError("pilot config hash mismatch; copy config.json unchanged, even its old run_dir")
    if (state["optimizer_step"], state["tokens_seen"], state["sequence_in_block"]) != (3072, 50331648, 0):
        raise ValueError("expected completed pilot at step 3072 / 50331648 tokens / safe block boundary")
    # Operational cadence and destination may differ; training semantics may not.
    ignored = {"run_dir", "log_every_steps", "checkpoint_every_steps", "validation_every_steps", "validation_batches"}
    if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in config.items() if k not in ignored}:
        raise ValueError("pilot training settings differ from the continuation config")
    args += ["--import-checkpoint", str(checkpoint)]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="validate import metadata and print command only")
    args = parser.parse_args()
    try:
        command = prepare_command(args.pilot_run, args.device, args.dry_run)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print("CUDARC_CUDA_VERSION=12090 " + shlex.join(command), flush=True)
    if not args.dry_run:
        # CUDA Toolkit 12.9 including NVRTC must be installed as documented.
        # Every invocation explicitly selects FP32, never cuda-fp16.
        raise SystemExit(subprocess.call(command, env=dict(os.environ, CUDARC_CUDA_VERSION="12090")))


if __name__ == "__main__":
    main()
