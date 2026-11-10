"""LogiQA 2.0 (official English MRC) and ARC-Challenge adapters."""

import random
from pathlib import Path

from baseline.pessimism.datasets.base_problem import SpecialCompletionDataset, SpecialCompletionProblem
from baseline.pessimism.datasets.choice_answers import (
    CHOICE_INSTRUCTION, extract_choice_answer, normalize_choice_row,
)


class ReasoningChoiceProblem(SpecialCompletionProblem):
    def check_correctness(self, response):
        return extract_choice_answer(response, self.extra["choice_labels"]) == self.answer


class ReasoningChoiceDataset(SpecialCompletionDataset):
    def __init__(self, dataset_name, split="test", seed=2, data_dir=None,
                 revision="main", shuffle_before_limit=False, limit_problems=None,
                 records=None, **kwargs):
        super().__init__(**kwargs)
        if dataset_name not in ("logiqa2", "arc_challenge"):
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        self.name_or_path = dataset_name
        if records is None:
            from datasets import load_dataset
            if dataset_name == "logiqa2":
                filename = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}[split]
                source = str(Path(data_dir) / filename) if data_dir else (
                    f"https://raw.githubusercontent.com/csitfun/LogiQA2.0/{revision}/"
                    f"logiqa/DATA/LOGIQA/{filename}"
                )
                records = load_dataset("json", data_files={split: source}, split=split)
            else:
                records = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, revision=revision)
        self.problems = []
        for index, row in enumerate(records):
            text, answer, extra = normalize_choice_row(dataset_name, row)
            extra.update(original_index=index, dataset=dataset_name, split=split, revision=revision)
            problem = ReasoningChoiceProblem(text, answer, extra=extra)
            # Set explicitly to bypass the legacy multiple-choice prompt branch.
            problem.prompt = CHOICE_INSTRUCTION + "\n\n" + text
            self.problems.append(problem)
        if shuffle_before_limit:
            random.Random(seed).shuffle(self.problems)
        if limit_problems is not None:
            if limit_problems <= 0:
                raise ValueError("limit_problems must be positive")
            self.problems = self.problems[:limit_problems]
