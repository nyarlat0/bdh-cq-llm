"""CPU-only harness checks: no CUDA, training, or dataset access."""
import json
from pathlib import Path
import unittest

from benchmark_v100_sizes import make_config, parse_result


class SizeSweepTests(unittest.TestCase):
    def test_shapes_and_budget(self):
        base = json.loads(Path("configs/v100-v2.json").read_text())
        before = json.dumps(base)
        for dim in (512, 640, 768, 1024):
            for batch in (1, 2, 4, 8, 16, 32):
                cfg = make_config(base, dim, batch, "test-only")
                self.assertEqual(cfg["model"]["dim_qk_heads"], dim * 12)
                self.assertEqual(cfg["model"]["rotary_dim"] * 16, dim * 12)
                for key, value in base["model"].items():
                    if key not in ("dim", "dim_qk_heads", "rotary_dim"):
                        self.assertEqual(cfg["model"][key], value)
                self.assertEqual(cfg["optimizer"]["gradient_accumulation"] * batch, 64)
                self.assertEqual(cfg["optimizer"]["gradient_accumulation"] % 2, 0)
                self.assertEqual(cfg["schedule_batch_multiple"], 32)
                self.assertEqual(cfg["memory"]["chunks_per_detach"], 2)
        self.assertEqual(json.dumps(base), before)

    def test_success_and_warmup(self):
        log = "model parameters: 22043648\n" + "\n".join(
            f"step {step} | General | loss 8.2 | {speed} tok/s"
            for step, speed in [(4, 100), (8, 200), (12, 4000), (16, 4200)])
        log += "\nrequested --max-steps reached at a safe block boundary"
        result = parse_result(log, 0, 8)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["median_tok_s"], 4100)
        self.assertEqual(result["parameters"], 22043648)
        for suffix, status, timeout, reason in [
            ("\nnon-finite loss NaN", 1, False, "non_finite"),
            ("\ncan't allocate buffer of size: 2129534976", 1, False, "allocation_failure"),
            ("", None, True, "timeout"),
        ]:
            failed = parse_result(log + suffix, status, 8, timeout)
            self.assertFalse(failed["eligible"])
            self.assertIsNone(failed["median_tok_s"])
            self.assertEqual(failed["reason"], reason)
        self.assertFalse(parse_result(log + "\nnon-finite loss NaN", 1, 8)["finite"])


if __name__ == "__main__":
    unittest.main()
