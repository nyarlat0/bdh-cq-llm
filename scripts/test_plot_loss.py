import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from plot_loss import read_loss
from plot_v2_pilot_results import svg_chart


class PlotTests(unittest.TestCase):
    def test_partial_json_and_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text('\n'.join([
                '{"event":"train","step":16,"tokens_seen":100,"loss":8}',
                '{"event":"train","step":32,"tokens_seen":200,"loss":7}',
                '{"event":"validation","step":32,"stateful_loss":7}',
                '{"event":"train","step":16,"tokens_seen":100,"loss":9}',
                '{"event":']))
            result = read_loss(path)
            self.assertEqual(result["train"], {16: (100, 9)})
            self.assertEqual(result["stateful"], {})
            path.write_text('{broken}\n{"event":"train"}\n')
            with self.assertRaises(ValueError):
                read_loss(path)

    def test_console_and_single_point_svg(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "console.log"
            path.write_text('step 16 | General | tokens 262144 | loss 8.0 | lr 0.01\n'
                            'validation step 16: memoryless 8.1, stateful Some(7.9), best 7.9\n'
                            'step 32 | General | tokens 524288 | loss NaN | lr 0.01\n')
            result = read_loss(path)
            self.assertEqual(result["stateful"][16], (262144, 7.9))
            self.assertNotIn(32, result["train"])
            output = Path(temp) / "single.svg"
            svg_chart({"test": ("red", [(16, 7.9)])}, "A&B", "one point", "loss", output, "steps")
            ET.parse(output)
            self.assertIn("A&amp;B", output.read_text())


if __name__ == "__main__":
    unittest.main()
