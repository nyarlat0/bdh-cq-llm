import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_v100_from_pilot as launcher


class ContinuationTests(unittest.TestCase):
    def test_import_then_resume_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads(launcher.CONFIG.read_text())
            cfg["run_dir"] = str(root / "production")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg))
            pilot = root / "pilot"
            checkpoint = pilot / "checkpoints/step-000000003072"
            checkpoint.mkdir(parents=True)
            previous = dict(cfg, run_dir="/old/machine/pilot", validation_batches=384)
            frozen = json.dumps(previous).encode()
            (pilot / "config.json").write_bytes(frozen)
            (checkpoint / "state.json").write_text(json.dumps(dict(
                optimizer_step=3072, tokens_seen=50331648, sequence_in_block=0,
                config_sha256=hashlib.sha256(frozen).hexdigest())))
            for name in ("model.bin", "optimizer.bin"):
                (checkpoint / name).touch()
            with patch.object(launcher, "CONFIG", config_path):
                command = launcher.launch_command(pilot, 0)
                self.assertIn("--import-checkpoint", command)
                self.assertNotIn("--max-steps", command)
                launcher.prepare_command(pilot, 0, dry_run=True)
                self.assertFalse((root / "production").exists())
                copied = launcher.prepare_command(pilot, 0)
                snapshot = root / "production/pilot-source"
                self.assertEqual(Path(copied[-1]), snapshot / "checkpoints/step-000000003072")
                self.assertEqual((snapshot / "config.json").read_bytes(), frozen)
                self.assertEqual(launcher.prepare_command(root / "missing", 0), copied)
                (snapshot / "checkpoints/step-000000003072/model.bin").write_text("corrupt")
                with self.assertRaisesRegex(ValueError, "corrupted"):
                    launcher.prepare_command(pilot, 0)
                (pilot / "config.json").write_bytes(frozen + b" ")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    launcher.launch_command(pilot, 0)
                destination = root / "production/checkpoints"
                destination.mkdir(parents=True)
                (destination / "latest.json").touch()
                self.assertNotIn("--import-checkpoint", launcher.launch_command(root / "missing", 0))
                (destination.parent / "STOP").touch()
                with self.assertRaisesRegex(ValueError, "STOP"):
                    launcher.launch_command(pilot, 0)


if __name__ == "__main__":
    unittest.main()
