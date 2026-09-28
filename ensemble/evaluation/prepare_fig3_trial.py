#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from baseline.pessimism.datasets.mathgsm import (
    MathGSMDataset,
)

from baseline.pessimism.datasets.bbh import (
    BBHDataset,
)


def build_prompt_config():

    return {
        "system_prompt": (
            "Solve the following problem step by step. "
            "Give your final numerical answer at the end with: "
            "#### {NUM}"
        ),
        "add_generation_prompt": True,
    }


def build_dataset(dataset_name):

    prompt_config = build_prompt_config()

    # ========================================================
    # MATH-500
    # ========================================================

    if dataset_name == "math500":

        print(
            "Reconstructing MATH-500 dataset..."
        )

        dataset = MathGSMDataset(
            seed=2,
            split="test",
            fewshot_num=0,
            math_split="math_500_test",
            limit_problems=500,
            prompt_postprocessor_config=(
                prompt_config
            ),
        )

        return dataset

    # ========================================================
    # BBH
    # ========================================================

    if dataset_name == "bbh":

        print(
            "Reconstructing complete BBH dataset..."
        )

        # We intentionally load all BBH problems.
        #
        # The original Figure 3 trial used a 1000-problem
        # subset. UUID matching below selects exactly those
        # problems appearing in detailed_candidates.json.
        #
        # Therefore we do not need to reproduce the random
        # subset ordering here.

        dataset = BBHDataset(
            seed=1,
            split="test",
            fewshot_num=0,
            limit_problems=None,
            shuffle_before_limit=False,
            prompt_postprocessor_config=(
                prompt_config
            ),
        )

        return dataset

    raise ValueError(
        f"Unsupported dataset: {dataset_name}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "math500",
            "bbh",
        ],
    )

    parser.add_argument(
        "--detailed",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load detailed candidates
    # ========================================================

    print(
        f"Loading detailed candidates: "
        f"{args.detailed}"
    )

    with open(
        args.detailed,
        "r",
        encoding="utf-8",
    ) as f:
        detailed = json.load(f)

    print(
        f"Detailed problems: "
        f"{len(detailed)}"
    )

    # ========================================================
    # 2. Reconstruct source dataset
    # ========================================================

    dataset = build_dataset(
        args.dataset
    )

    print(
        f"Reconstructed problems: "
        f"{len(dataset)}"
    )

    # ========================================================
    # 3. UUID mapping
    # ========================================================

    problem_by_uuid = {
        str(problem.uuid): problem
        for problem in dataset
    }

    detailed_ids = set(
        str(x)
        for x in detailed.keys()
    )

    dataset_ids = set(
        problem_by_uuid.keys()
    )

    matched = (
        detailed_ids
        & dataset_ids
    )

    missing = (
        detailed_ids
        - dataset_ids
    )

    print()
    print("=" * 60)
    print("UUID matching")
    print("=" * 60)

    print(
        f"Matched: "
        f"{len(matched)}"
    )

    print(
        f"Missing: "
        f"{len(missing)}"
    )

    if missing:

        print(
            "Example missing IDs:",
            list(missing)[:5],
        )

        raise RuntimeError(
            "Could not reconstruct all "
            "evaluation problems."
        )

    # ========================================================
    # 4. Validate candidates
    # ========================================================

    for problem_id, item in (
        detailed.items()
    ):

        responses = item[
            "responses"
        ]

        if len(responses) != 512:

            raise RuntimeError(
                f"{problem_id}: expected "
                f"512 candidates, "
                f"got {len(responses)}"
            )

        expected_keys = {
            str(i)
            for i in range(
                1,
                513,
            )
        }

        actual_keys = set(
            responses.keys()
        )

        if actual_keys != expected_keys:

            raise RuntimeError(
                f"{problem_id}: response "
                "keys are not exactly 1..512"
            )

    print()
    print(
        "Candidate validation PASS"
    )

    # ========================================================
    # 5. Write generic NeuBoots scorer input
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as out:

        for (
            problem_id,
            item,
        ) in detailed.items():

            problem = (
                problem_by_uuid[
                    str(problem_id)
                ]
            )

            responses = item[
                "responses"
            ]

            samples = [
                responses[str(i)][
                    "content"
                ]
                for i in range(
                    1,
                    513,
                )
            ]

            row = {
                "instance_id":
                    str(problem_id),

                "prompt":
                    problem.prompt,

                "all_samples":
                    samples,
            }

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        f"Dataset: "
        f"{args.dataset}"
    )

    print(
        f"Problems: "
        f"{len(detailed)}"
    )

    print(
        "Candidates/problem: 512"
    )

    print(
        f"Saved: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()