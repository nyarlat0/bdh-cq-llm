"""Resume integration using a fake trainer: never starts CUDA or training."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import pilot_v100_sizes as pilot


def save_checkpoint(run, step):
    checkpoint = run / "checkpoints" / f"step-{step:012}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "state.json").write_text(json.dumps({"optimizer_step": step}))
    (checkpoint.parent / "latest.json").write_text(json.dumps({
        "optimizer_step": step, "checkpoint_dir": checkpoint.name}))


class ResumeTests(unittest.TestCase):
    def test_rollback_and_partial_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text('{"step": 16}\n{"step": 32}\n{"ste')
            pilot.prune_events(path, 16)
            self.assertEqual(pilot.load_events(path), [{"step": 16}])
            self.assertTrue(list(Path(temp).glob("train-before-resume-*.jsonl")))
            curve = Path(temp) / "validation.csv"
            curve.write_text("dim,step\n512,16\n512,32\n640,32\n")
            pilot.prune_curve(curve, 512, 16)
            self.assertNotIn("512,32", curve.read_text())
            self.assertIn("640,32", curve.read_text())

    def test_legacy_resume_remaining_budget_and_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = root / "tokenizer.json"
            tokenizer.touch()
            base = json.loads(Path("configs/v100-v2.json").read_text())
            base.update(tokenizer=str(tokenizer), packed_dir=str(root))
            for source in base["sources"]:
                (root / f"{source}.tokens").touch()
            (root / "train-fp32").touch()
            for index, (dim, batch) in enumerate(pilot.CASES):
                label = f"fp32-d{dim}-b{batch}"
                run = root / label
                cfg = pilot.pilot_config(base, dim, batch, run, 256)
                (root / f"{label}.json").write_text(json.dumps(cfg))
                step = 256 if index < 2 else 512
                save_checkpoint(run, step)
                (run / "train.jsonl").write_text(json.dumps(dict(
                    event="validation", step=256, selected_loss=7.0)) + "\n")
                (root / f"{label}.log").write_text("model parameters: 123\n")
            (root / "STOP").touch()
            (root / "fp32-d512-b8/STOP").touch()
            calls = []

            def fake_process(args, **kwargs):
                calls.append(args)
                self.assertEqual(args[-1], "256")  # NOT another 512 updates
                cfg = json.loads(Path(args[args.index("--config") + 1]).read_text())
                run = Path(cfg["run_dir"])
                save_checkpoint(run, 512)
                # Real trainer skips validation at --max-steps: reproduce that
                # protocol, rather than inventing a final validation in the mock.
                log = ("model parameters: 123\nstep 304 | loss 7.0 | 4000 tok/s\n"
                       "requested --max-steps reached at a safe block boundary\n")
                return SimpleNamespace(stdout=io.StringIO(log), wait=lambda: 0)

            argv = ["pilot", "--resume", str(root), "--updates", "512"]
            with patch("sys.argv", argv), patch.object(pilot, "read_gpu", return_value=("uuid", "V100")), \
                 patch.object(pilot, "command", return_value=SimpleNamespace(stdout="release 12.9")), \
                 patch.object(pilot.subprocess, "Popen", side_effect=fake_process), patch("sys.stdout", io.StringIO()):
                pilot.main()
                self.assertEqual(len(calls), 2)
                # Saved budget now suffices; all four completed cases skipped.
                with patch("sys.argv", ["pilot", "--resume", str(root)]):
                    pilot.main()
                self.assertEqual(len(calls), 2)
            self.assertFalse((root / "STOP").exists())
            self.assertTrue(list(root.glob("STOP.acknowledged-*")))
            self.assertEqual(json.loads((root / "pilot.json").read_text())["updates"], 512)
            self.assertEqual(len(json.loads((root / "results.json").read_text())), 4)
            for row in json.loads((root / "results.json").read_text()):
                self.assertTrue(row["completed"])
                self.assertTrue(row["eligible"])
                self.assertEqual(row["last_validation_step"], 256)
                self.assertFalse(row["final_validation_present"])


if __name__ == "__main__":
    unittest.main()
