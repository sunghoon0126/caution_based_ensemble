import math
import unittest

from baseline.pessimism.models.generation_logprobs import (
    summarize_choice_logprobs,
)
from ensemble.generation.generate_reference_candidates import (
    canonicalize_record,
)


class SummarizeChoiceLogprobsTest(unittest.TestCase):
    def test_computes_sequence_and_average_nll(self):
        choice = {
            "logprobs": {
                "content": [
                    {
                        "token": "A",
                        "bytes": [65],
                        "logprob": math.log(0.5),
                    },
                    {
                        "token": "B",
                        "bytes": [66],
                        "logprob": math.log(0.25),
                    },
                ]
            }
        }

        result = summarize_choice_logprobs(choice)

        self.assertEqual(result["token_count"], 2)
        self.assertAlmostEqual(result["sequence_probability"], 0.125)
        self.assertAlmostEqual(
            result["sequence_nll"],
            -math.log(0.125),
        )
        self.assertAlmostEqual(
            result["average_nll"],
            result["sequence_nll"] / 2,
        )
        self.assertAlmostEqual(
            result["geometric_mean_token_probability"],
            math.sqrt(0.125),
        )

    def test_requires_logprobs(self):
        with self.assertRaisesRegex(ValueError, "logprobs"):
            summarize_choice_logprobs({})

    def test_rejects_empty_token_logprobs(self):
        with self.assertRaisesRegex(ValueError, "no chosen-token"):
            summarize_choice_logprobs(
                {"logprobs": {"content": []}}
            )

    def test_long_sequence_probability_can_underflow(self):
        result = summarize_choice_logprobs(
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "x",
                            "bytes": [120],
                            "logprob": -1000.0,
                        }
                    ]
                }
            }
        )

        self.assertEqual(result["sequence_probability"], 0.0)
        self.assertEqual(result["sequence_nll"], 1000.0)


class CanonicalizeRecordTest(unittest.TestCase):
    def test_records_identity_correctness_and_likelihood(self):
        class FakeProblem:
            def check_correctness(self, response):
                return response == "answer"

        raw_record = {
            "request": {
                "candidate_id": "candidate-0",
                "dataset": "gsm8k",
                "split": "test",
                "trial_id": 1,
                "dataset_index": 3,
                "instance_id": "problem-3",
                "sample_index": 0,
                "prompt": "prompt",
                "messages": [{"role": "user", "content": "prompt"}],
                "problem": "problem",
                "ground_truth": "answer",
                "generation_config": {
                    "temperature": 1.0,
                    "logprobs": True,
                },
            },
            "response": {
                "model": "reference-model",
                "created": 123,
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "answer",
                                    "bytes": None,
                                    "logprob": math.log(0.8),
                                }
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        }

        result = canonicalize_record(
            raw_record,
            {"problem-3": FakeProblem()},
        )

        self.assertTrue(result["correct"])
        self.assertEqual(result["trial_id"], 1)
        self.assertEqual(result["token_count"], 1)
        self.assertAlmostEqual(result["sequence_probability"], 0.8)


if __name__ == "__main__":
    unittest.main()
