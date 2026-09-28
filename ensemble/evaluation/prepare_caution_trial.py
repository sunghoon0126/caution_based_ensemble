import argparse
import json
from pathlib import Path

from baseline.pessimism.datasets.gsm8k import (
    GSM8KDataset,
)


def main():
    parser = argparse.ArgumentParser()

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
    # 1. Load Caution detailed candidates
    # ========================================================

    print(
        f"Loading detailed: "
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
    # 2. Reconstruct GSM8K test dataset
    #
    # UUID is determined from the original problem,
    # answer, generation_config, and extra fields.
    # ========================================================

    prompt_postprocessor_config = {
        "system_prompt": (
            "Solve the following problem step by step. "
            "Give your final numerical answer at the end with: "
            "#### {NUM}"
        ),
        "add_generation_prompt": True,
    }

    print()
    print("Loading GSM8K test dataset...")

    dataset = GSM8KDataset(
        seed=2,
        split="test",
        name_or_path="gsm8k",
        config_name="main",
        fewshot_num=0,
        limit_problems=1000,
        prompt_postprocessor_config=(
            prompt_postprocessor_config
        ),
    )

    print(
        f"GSM8K problems: "
        f"{len(dataset)}"
    )

    # ========================================================
    # 3. Build UUID -> problem mapping
    # ========================================================

    problem_by_uuid = {
        problem.uuid: problem
        for problem in dataset
    }

    detailed_ids = set(
        detailed.keys()
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

    extra = (
        dataset_ids
        - detailed_ids
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
        f"Missing from dataset: "
        f"{len(missing)}"
    )

    print(
        f"Extra dataset problems: "
        f"{len(extra)}"
    )

    if missing:
        print(
            "Example missing UUIDs:",
            list(missing)[:5],
        )

    if len(matched) != len(
        detailed
    ):
        raise RuntimeError(
            "Could not match all detailed "
            "problems to GSM8K dataset."
        )

    # ========================================================
    # 4. Validate detailed candidates
    # ========================================================

    for problem_id, item in (
        detailed.items()
    ):
        responses = item[
            "responses"
        ]

        if len(responses) != 512:
            raise RuntimeError(
                f"{problem_id}: "
                f"expected 512 responses, "
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

        if (
            actual_keys
            != expected_keys
        ):
            raise RuntimeError(
                f"{problem_id}: "
                "response keys are not 1..512"
            )

    print()
    print(
        "Candidate validation PASS: "
        "512 responses per problem"
    )

    # ========================================================
    # 5. Build NeuBoots scorer input
    #
    # Candidate ordering:
    # detailed responses["1"] ... responses["512"]
    #
    # This is exactly the ordering later used by
    # evaluate_fig3_methods.py.
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

        for problem_id, item in (
            detailed.items()
        ):

            problem = (
                problem_by_uuid[
                    problem_id
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
                    problem_id,

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
        f"Problems: "
        f"{len(detailed)}"
    )

    print(
        "Candidates/problem: "
        "512"
    )

    print(
        f"Saved: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()