import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ensemble.evaluation.baseline6.common import ResumeWriter, digest, groups
from ensemble.evaluation.baseline6.evaluate_bon import evaluate
from ensemble.evaluation.baseline6.score_dummy_backprop import run


def source_rows():
    return [dict(dataset="gsm8k", split="test", trial_id=1, dataset_index=0,
                 instance_id="p0", candidate_id=f"c{i}", sample_index=i, prompt="p", response=str(i),
                 correct=i == 1, reward_model_score=reward, reward_model_path="fake",
                 reward_model_revision="pinned", model="generator",
                 reward_scoring_config=dict(backend="pair", dtype="float32", revision="main", max_length=None))
            for i, reward in enumerate([2.5, 0.8, -1.2, 3.1])]


def save(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class PipelineTest(unittest.TestCase):
    def test_fixed_full_pool_resume_and_bon(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "original.jsonl"
            output = Path(folder) / "bp.jsonl"
            source = source_rows()
            save(original, source)
            calls = []
            class FakeAdapter:
                settings = source[0]["reward_scoring_config"]
                model = SimpleNamespace(config=SimpleNamespace(_commit_hash="pinned"))
                def __init__(self, *args):
                    pass
                def adapted_score(self, prompt, response, target, lr, steps, scope):
                    calls.append(target)
                    before = source[int(response)]["reward_model_score"]
                    return dict(reward_before=before, reward_after=10.0 if response == "1" else target,
                                reward_change=0, gradient_norm=0, dummy_loss_before=0, dummy_loss_after=0)
            args = SimpleNamespace(input_path=original, output_path=output, reward_model="fake", num_samples=4,
                                   learning_rate=0.01, num_sgd_steps=1, update_scope="full", device="cpu",
                                   reward_backend="pair", dtype="float32", revision="main", max_length=None,
                                   before_rtol=1e-4, before_atol=1e-4)
            with patch("ensemble.evaluation.baseline6.score_dummy_backprop.RewardModelAdapter", FakeAdapter):
                run(args)
                self.assertEqual(calls, [-1.2] * 4)
                run(args)
                self.assertEqual(len(calls), 4)
                curves, summary, metadata = evaluate([original], [output], 4)
                n1 = [r["accuracy_percent"] for r in curves if r["N"] == 1]
                self.assertEqual(n1, [0, 0])
                self.assertEqual(summary[1]["final_accuracy_percent"], 100)
                self.assertEqual(metadata["dummy_calibration_pool"], 4)
                args.learning_rate = 0.02
                with self.assertRaisesRegex(ValueError, "Resume input/config mismatch"):
                    run(args)
                args.learning_rate = 0.01
                source[-1]["reward_model_score"] = -2.0
                save(original, source)
                with self.assertRaisesRegex(ValueError, "Resume input/config mismatch"):
                    run(args)

    def test_group_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.jsonl"
            save(path, source_rows()[:3])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                list(groups(path, 4))
            data = source_rows()
            data[1]["candidate_id"] = data[0]["candidate_id"]
            save(path, data)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                list(groups(path, 4))

    def test_resume_rejects_truncated_rows_and_output_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.jsonl"
            path.write_text('{"candidate_id":')
            with self.assertRaisesRegex(ValueError, "Truncated"):
                ResumeWriter(path, {})
            path.write_text('{}\n')
            writer = ResumeWriter(path, {})
            with self.assertRaisesRegex(ValueError, "more candidates"):
                writer.finish()
            writer.close()

    def test_evaluation_tie_first_n_and_trial_average(self):
        with tempfile.TemporaryDirectory() as folder:
            originals, adapted = [], []
            for trial in (1, 2):
                source = source_rows()
                for row in source:
                    row["trial_id"] = trial
                dummy = -1.2
                context = dict(dummy_reward=dummy, pool_hash=digest(source))
                bp = [dict(row, reward_after=0.0, finite=True, dummy_reward=dummy,
                           dummy_calibration_pool=4, backprop_config=dict(learning_rate=0.1),
                           input_fingerprint=digest({"source": row, "context": context})) for row in source]
                op, ap = Path(folder) / f"o{trial}.jsonl", Path(folder) / f"a{trial}.jsonl"
                save(op, source)
                save(ap, bp)
                originals.append(op)
                adapted.append(ap)
            curves, summaries, meta = evaluate(originals, adapted, 4)
            self.assertTrue(all(r["accuracy_percent"] == 0 for r in curves if r["method"] == "dummy_backprop_rm"))
            self.assertEqual(meta["trial_ids"], [1, 2])
            bp[0]["reward_after"] = float("nan")
            save(adapted[-1], bp)
            with self.assertRaises(FloatingPointError):
                evaluate(originals, adapted, 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; run these actual autograd tests in server environment")
class FunctionalSGDTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.model = torch.nn.Linear(2, 1).double()
        with torch.no_grad():
            self.model.weight.fill_(0.5)
            self.model.bias.fill_(0.2)
        self.a = dict(input=torch.tensor([[1.0, 2.0]], dtype=torch.float64))
        self.b = dict(input=torch.tensor([[2.0, -1.0]], dtype=torch.float64))

    def adapt(self, inputs, target=0.0, lr=0.01, steps=1):
        from ensemble.evaluation.baseline6.reward_model_adapter import functional_adapt
        return functional_adapt(self.model, inputs, target, lr, steps)

    def test_actual_update_matches_linear_sgd_and_model_unchanged(self):
        state = {k: p.detach().clone() for k, p in self.model.state_dict().items()}
        result = self.adapt(self.a)
        # r' = r - eta * r * (||x||^2 + 1) for this linear model.
        self.assertAlmostEqual(result["reward_after"], 1.7 * (1 - 0.01 * 6), places=5)
        for name, param in self.model.state_dict().items():
            self.assertTrue(self.torch.equal(param, state[name]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_zero_steps_and_zero_loss(self):
        result = self.adapt(self.a, steps=0)
        self.assertEqual(result["reward_before"], result["reward_after"])
        result = self.adapt(self.a, target=result["reward_before"])
        self.assertEqual(result["reward_before"], result["reward_after"])
        self.assertEqual(result["gradient_norm"], 0)

    def test_order_independence_and_multiple_steps(self):
        a, b = self.adapt(self.a, steps=2), self.adapt(self.b, steps=2)
        self.assertEqual(b, self.adapt(self.b, steps=2))
        self.assertEqual(a, self.adapt(self.a, steps=2))
        self.assertLess(a["reward_after"], a["reward_before"])

    def test_overflow_is_rejected(self):
        with self.assertRaises((FloatingPointError, RuntimeError)):
            self.adapt(self.a, lr=1e308, steps=2)


if __name__ == "__main__":
    unittest.main()
