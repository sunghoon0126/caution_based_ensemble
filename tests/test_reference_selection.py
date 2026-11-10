import json
import tempfile
import unittest
from pathlib import Path

from ensemble.evaluation.prepare_reference_selection import calibrate, tune


class HeldOutSelectionTest(unittest.TestCase):
    def test_train_calibration_validation_tuning_and_test_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.jsonl"
            def write(split, model="rm"):
                rows = [dict(dataset="logiqa2", split=split, trial_id=1, dataset_index=0,
                             instance_id="problem", sample_index=i, reward_model_path=model,
                             reward_model_score=2 - i, sequence_nll=10 - 9 * i,
                             average_nll=2 - i, correct=bool(i)) for i in range(2)]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write("train")
            stats = calibrate([path])
            self.assertEqual(stats["reward"]["mean"], 1.5)
            write("validation")
            selected = tune([path], stats, [0, 2], 2)
            self.assertEqual(selected["seq_lambda"], 2)
            self.assertEqual(selected["avg_lambda"], 2)
            write("validation", "other_rm")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                tune([path], stats, [0, 2], 2)
            write("test")
            with self.assertRaisesRegex(ValueError, "Expected train"):
                calibrate([path])
            with self.assertRaisesRegex(ValueError, "Expected validation"):
                tune([path], stats, [0, 2], 2)


if __name__ == "__main__":
    unittest.main()
