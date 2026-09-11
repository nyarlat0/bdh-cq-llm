#!/usr/bin/env python3

"""
Continue the completed 63M V100 size pilot into the full ~1.05B-token run.

Preserves:
- model weights;
- AdamW state;
- data/schedule cursor;
- architecture;
- optimizer semantics;
- CQ semantics used by the pilot.

Only operational cadence and destination run_dir are changed.
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


CONFIG = Path("configs/v100-63m-from-pilot.json")
DEST_RUN = Path("runs/v100-63m-cq-from-pilot")

DEFAULT_PILOT = Path(
    "runs/v100-size-pilots-20260908-131245-087810/fp32-d1024-b4"
)

PILOT_STEP = 3072
PILOT_TOKENS = 50_331_648

CHECKPOINT_NAME = f"step-{PILOT_STEP:012d}"

IMPORT_FILES = (
    "config.json",
    f"checkpoints/{CHECKPOINT_NAME}/state.json",
    f"checkpoints/{CHECKPOINT_NAME}/model.bin",
    f"checkpoints/{CHECKPOINT_NAME}/optimizer.bin",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_pilot(pilot: Path) -> dict:
    config_path = pilot / "config.json"
    checkpoint = pilot / "checkpoints" / CHECKPOINT_NAME
    state_path = checkpoint / "state.json"

    for path in (
        config_path,
        state_path,
        checkpoint / "model.bin",
        checkpoint / "optimizer.bin",
    ):
        if not path.is_file():
            raise ValueError(f"missing pilot file: {path}")

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    state = load_json(state_path)

    if hashlib.sha256(config_bytes).hexdigest() != state["config_sha256"]:
        raise ValueError("pilot config.json SHA-256 does not match state.json")

    expected_state = (
        PILOT_STEP,
        PILOT_TOKENS,
        0,
    )
    actual_state = (
        state["optimizer_step"],
        state["tokens_seen"],
        state["sequence_in_block"],
    )

    if actual_state != expected_state:
        raise ValueError(
            f"pilot is not the expected completed safe checkpoint: "
            f"got {actual_state}, expected {expected_state}"
        )

    model = config["model"]
    expected_model = {
        "dim": 1024,
        "depth": 8,
        "heads": 8,
        "dim_qk_heads": 12288,
        "rotary_dim": 768,
    }

    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(
                f"unexpected 63M architecture: model.{key}="
                f"{model.get(key)}, expected {expected}"
            )

    optimizer = config["optimizer"]
    if optimizer["micro_batch_size"] != 4:
        raise ValueError("63M pilot must use micro_batch_size=4")

    if optimizer["gradient_accumulation"] != 16:
        raise ValueError("63M pilot must use gradient_accumulation=16")

    memory = config["memory"]

    # The size pilot trained with CQ fully active from token zero.
    # Continue the SAME trajectory instead of silently changing objective.
    if memory["stateful_after_tokens"] != 0:
        raise ValueError("pilot must have CQ active from token zero")

    if memory["memory_read_ramp_tokens"] != 0:
        raise ValueError("pilot must have no CQ read ramp")

    if memory["chunks_per_detach"] != 2:
        raise ValueError("pilot must use TBPTT=2 chunks")

    if config.get("schedule_batch_multiple") != 32:
        raise ValueError("pilot must use schedule_batch_multiple=32")

    return config


def continuation_config(pilot_config: dict) -> dict:
    # Deep-copy using JSON because the config is JSON-native anyway.
    config = json.loads(json.dumps(pilot_config))

    config["run_dir"] = str(DEST_RUN)

    # Operational-only settings. Training semantics remain untouched.
    config["log_every_steps"] = 10
    config["checkpoint_every_steps"] = 625
    config["validation_every_steps"] = 625
    config["validation_batches"] = 1536

    return config


def ensure_config(config: dict, write: bool):
    text = json.dumps(config, indent=2) + "\n"

    if CONFIG.exists():
        current = load_json(CONFIG)
        if current != config:
            raise ValueError(
                f"{CONFIG} already exists but differs from the expected "
                "63M continuation config"
            )
        return

    if write:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(text)
        print(f"created {CONFIG}", flush=True)
    else:
        print(f"would create {CONFIG}", flush=True)


def verify_snapshot(snapshot: Path):
    manifest_path = snapshot / "copy-sha256.json"

    if not manifest_path.is_file():
        raise ValueError(f"{snapshot} exists but has no copy-sha256.json")

    manifest = load_json(manifest_path)

    if set(manifest) != set(IMPORT_FILES):
        raise ValueError("pilot snapshot manifest contains unexpected files")

    for name, expected in manifest.items():
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"pilot snapshot is incomplete: {path}")

        if digest(path) != expected:
            raise ValueError(f"pilot snapshot is corrupted: {path}")

    validate_pilot(snapshot)


def snapshot_pilot(pilot: Path) -> Path:
    snapshot = DEST_RUN / "pilot-source"

    if snapshot.exists():
        verify_snapshot(snapshot)
        return snapshot

    DEST_RUN.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=".pilot-source-copy-",
            dir=DEST_RUN,
        )
    )

    manifest = {}

    try:
        for name in IMPORT_FILES:
            source = pilot / name
            destination = staging / name

            expected = digest(source)

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

            if digest(destination) != expected:
                raise ValueError(
                    f"copy verification failed for {name}; "
                    f"staging retained at {staging}"
                )

            manifest[name] = expected

        (staging / "copy-sha256.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        validate_pilot(staging)

        staging.rename(snapshot)

        print(
            f"copied and SHA-256 verified 63M pilot into {snapshot}",
            flush=True,
        )

        return snapshot

    except Exception:
        print(f"failed staging retained at {staging}", flush=True)
        raise


def command(device: int, import_source: Path | None):
    args = [
        "cargo",
        "run",
        "--release",
        "--locked",
        "--no-default-features",
        "--features",
        "cuda",
        "--bin",
        "train_llm",
        "--",
        "--config",
        str(CONFIG),
        "--device",
        str(device),
    ]

    if import_source is not None:
        checkpoint = (
            import_source
            / "checkpoints"
            / CHECKPOINT_NAME
        )

        args += [
            "--import-checkpoint",
            str(checkpoint),
        ]

    # Intentionally NO --max-steps.
    # The trainer therefore consumes the rest of the normal full schedule.
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--pilot-run",
        type=Path,
        default=DEFAULT_PILOT,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    latest = DEST_RUN / "checkpoints" / "latest.json"

    if (DEST_RUN / "STOP").exists():
        parser.error(
            f"{DEST_RUN / 'STOP'} exists; remove it explicitly before resume"
        )

    try:
        if latest.exists():
            # Already imported previously. Resume only from destination.
            if not CONFIG.is_file():
                raise ValueError(
                    f"{CONFIG} is missing but destination already has checkpoints"
                )

            saved_config = load_json(CONFIG)

            if Path(saved_config["run_dir"]) != DEST_RUN:
                raise ValueError(
                    f"{CONFIG} points to unexpected run_dir "
                    f"{saved_config['run_dir']}"
                )

            import_source = None

        else:
            pilot_config = validate_pilot(args.pilot_run)
            full_config = continuation_config(pilot_config)

            ensure_config(
                full_config,
                write=not args.dry_run,
            )

            if args.dry_run:
                import_source = args.pilot_run
            else:
                import_source = snapshot_pilot(args.pilot_run)

        cmd = command(args.device, import_source)

    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

    print(
        "CUDARC_CUDA_VERSION=12090 "
        + shlex.join(cmd),
        flush=True,
    )

    if args.dry_run:
        return

    env = dict(
        os.environ,
        CUDARC_CUDA_VERSION="12090",
    )

    raise SystemExit(
        subprocess.call(
            cmd,
            env=env,
        )
    )


if __name__ == "__main__":
    main()
