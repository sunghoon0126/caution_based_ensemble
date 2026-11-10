import io
from contextlib import nullcontext
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from baseline.pessimism.datasets.choice_answers import extract_choice_answer, normalize_choice_row
from ensemble.generation.generate_reference_candidates import resolve_max_tokens, select_problems, validate_cached_requests
from ensemble.models.scalar_reward_inputs import resolve_backend, tokenize_rewards
from ensemble.evaluation.score_reference_candidates import load_reward_model, match_completed_prefix, score_batch, write_batch


LOGIQA = dict(id=7, text="All cats are mammals.", question="What follows?",
              options=["Cats are mammals", "Cats are birds", "No cats exist", "All mammals are cats"], answer=0)
ARC = dict(id="science-1", question="Which is a mammal?",
           choices={"label": ["1", "2", "3"], "text": ["fish", "cat", "bird"]}, answerKey="2")


class ChoiceAnswersTest(unittest.TestCase):
    def test_final_line_only_and_available_choices(self):
        for response in ("A is tempting", "#### A because...", "#### E", "#### A\nActually B", "#### A or B", "#### AB"):
            with self.subTest(response=response):
                self.assertIsNone(extract_choice_answer(response, list("ABCD")))
        self.assertEqual(extract_choice_answer("Discuss A and C.\n#### B\n", list("ABCD")), "B")

    def test_source_specific_labels(self):
        text, answer, extra = normalize_choice_row("logiqa2", LOGIQA)
        self.assertEqual(answer, "A")
        self.assertIn("Context:", text)
        text, answer, extra = normalize_choice_row("arc_challenge", ARC)
        self.assertEqual(answer, "B")
        self.assertEqual(extra["source_choice_labels"], ["1", "2", "3"])
        self.assertIn("B. cat", text)
        lettered = dict(ARC, choices={"label": ["C", "A", "B"], "text": ["fish", "cat", "bird"]}, answerKey="A")
        self.assertEqual(normalize_choice_row("arc_challenge", lettered)[1], "B")

    def test_bad_ground_truth_fails_loudly(self):
        for row in (dict(LOGIQA, answer=-1), dict(LOGIQA, answer=4), dict(LOGIQA, answer="A")):
            with self.assertRaises(ValueError):
                normalize_choice_row("logiqa2", row)
        with self.assertRaises(ValueError):
            normalize_choice_row("arc_challenge", dict(ARC, answerKey="D"))

    def test_dataset_adapter_selection_and_correctness_without_downloads(self):
        # Only dependency imports are stubbed; exercise the real base/adapter classes.
        stubs = {name: types.ModuleType(name) for name in ("shortuuid", "transformers", "tqdm")}
        stubs["shortuuid"].uuid = lambda: "unused"
        stubs["transformers"].AutoTokenizer = Mock()
        stubs["tqdm"].tqdm = lambda iterable, **kwargs: iterable
        with patch.dict(sys.modules, stubs):
            from baseline.pessimism.datasets.reasoning_choice import ReasoningChoiceDataset
        rows = [dict(LOGIQA, id=i) for i in range(8)]
        dataset = ReasoningChoiceDataset("logiqa2", records=rows, split="validation")
        self.assertTrue(dataset[0].check_correctness("Reasoning.\n#### A"))
        self.assertFalse(dataset[0].check_correctness("A"))
        self.assertNotIn("{NUM}", dataset[0].prompt)
        self.assertEqual(dataset[0].answer, "A")
        self.assertEqual([i for i, _ in select_problems(dataset, 2, 3)], [2, 3, 4])
        a = ReasoningChoiceDataset("logiqa2", records=rows, seed=3, shuffle_before_limit=True)
        b = ReasoningChoiceDataset("logiqa2", records=rows, seed=3, shuffle_before_limit=True)
        self.assertEqual([p.uuid for p in a], [p.uuid for p in b])
        self.assertEqual(resolve_max_tokens("logiqa2"), 1024)
        self.assertEqual(resolve_max_tokens("arc_challenge"), 1024)
        loader = Mock(return_value=rows)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = loader
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            ReasoningChoiceDataset("logiqa2", split="validation", data_dir="/fixture")
            loader.assert_called_with("json", data_files={"validation": "/fixture/dev.txt"}, split="validation")
            loader.return_value = [ARC]
            ReasoningChoiceDataset("arc_challenge", split="train", revision="pinned")
            loader.assert_called_with("allenai/ai2_arc", "ARC-Challenge", split="train", revision="pinned")


class RewardInputsTest(unittest.TestCase):
    def test_loading_and_scalar_scores_without_model_weights(self):
        torch_stub = types.ModuleType("torch")
        torch_stub.bfloat16 = "bf16"
        torch_stub.inference_mode = nullcontext
        transformers_stub = types.ModuleType("transformers")
        tokenizer = Mock(pad_token="<pad>", pad_token_id=0, chat_template="template")
        model = MagicMock()
        model.to.return_value = model
        model.config.num_labels = 1
        model.config._commit_hash = "checkpoint-sha"
        transformers_stub.AutoTokenizer = Mock()
        transformers_stub.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers_stub.AutoModelForSequenceClassification = Mock()
        factory = transformers_stub.AutoModelForSequenceClassification.from_pretrained
        factory.return_value = model
        settings = dict(backend="skywork", dtype="bfloat16", max_length=None, revision="pinned")
        with patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}):
            load_reward_model("rm", "cuda:0", settings)
            factory.assert_called_once_with("rm", revision="pinned", torch_dtype="bf16", num_labels=1)
            self.assertEqual(tokenizer.padding_side, "right")
            values = model.return_value.logits.reshape.return_value
            values.shape = (2, 1)
            values.__getitem__.return_value.detach.return_value.cpu.return_value.tolist.return_value = [1.5, -2.0]
            with patch("ensemble.evaluation.score_reference_candidates.tokenize_rewards", return_value={}):
                self.assertEqual(score_batch(tokenizer, model, [{}, {}], "cuda:0"), [1.5, -2.0])
            record = dict(candidate_id="x", response="text", sequence_nll=4.0, average_nll=2.0)
            output = io.StringIO()
            with patch("ensemble.evaluation.score_reference_candidates.score_batch", return_value=[1.5]):
                write_batch(output, [record], tokenizer, model, "cuda:0", "rm", False)
            scored = json.loads(output.getvalue())
            self.assertEqual(scored["sequence_nll"], record["sequence_nll"])
            self.assertEqual(scored["average_nll"], record["average_nll"])
            self.assertEqual(scored["reward_model_score"], 1.5)
            self.assertNotIn("reward_model_score", record)

    def test_skywork_template_bos_and_batch(self):
        tokenizer = Mock()
        tokenizer.bos_token = "<bos>"
        tokenizer.apply_chat_template.return_value = "<bos>conversation"
        tokenizer.return_value = {"input_ids": [[1, 2], [1, 2, 3]]}
        records = [dict(prompt="question", response="answer")] * 2
        tokenize_rewards(tokenizer, records, "skywork")
        tokenizer.assert_called_once_with(["conversation", "conversation"], truncation=False, padding=False)
        conversation = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual([r["role"] for r in conversation], ["user", "assistant"])
        tokenizer.pad.assert_called_once()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            tokenize_rewards(tokenizer, records, "skywork", max_length=2)

    def test_pair_baseline_and_model_detection(self):
        tokenizer = Mock()
        tokenize_rewards(tokenizer, [dict(prompt="p", response="r")], "pair")
        tokenizer.assert_called_once_with(["p"], ["r"], padding=True, truncation=True, return_tensors="pt")
        self.assertEqual(resolve_backend("Skywork/Skywork-Reward-V2-Qwen3-1.7B"), "skywork")
        self.assertEqual(resolve_backend("local/checkpoint", "skywork"), "skywork")
        self.assertEqual(resolve_backend("OpenAssistant/reward-model-deberta-v3-large-v2"), "pair")

    def test_resume_rejects_changed_rm_settings(self):
        row = dict(candidate_id="x", response_sha256="hash", response="r", reward_model_path="rm",
                   reward_model_score=1.0, reward_scoring_config={"backend": "pair"})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "scores.jsonl"
            output.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "settings differ"):
                match_completed_prefix(io.StringIO(json.dumps(row) + "\n"), output, "rm", False, None,
                                       Path("input"), {"backend": "skywork"})

    def test_generation_resume_rejects_changed_token_limit(self):
        request = dict(uuid="x", candidate_id="x", messages=[], generation_config={"max_tokens": 500})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "x.json").write_text(json.dumps({"request": request}))
            validate_cached_requests(path, [request])
            with self.assertRaisesRegex(ValueError, "generation_config"):
                validate_cached_requests(path, [dict(request, generation_config={"max_tokens": 1024})])


if __name__ == "__main__":
    unittest.main()
