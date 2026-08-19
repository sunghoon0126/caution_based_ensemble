import os
import csv
import json
import argparse

import numpy as np


SUPPORTED_METHODS = [
    "rm",
    "pessimism",
    "caution",
    "neuboots",
]


def load_jsonl_by_id(path):
    data = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            instance_id = item[
                "instance_id"
            ]

            if instance_id in data:
                raise ValueError(
                    f"Duplicate instance_id: "
                    f"{instance_id}"
                )

            data[instance_id] = item

    return data


def running_selection(
        scores,
        accuracy,
):
    """
    For N = 1, ..., num_candidates,
    select the highest-scoring candidate
    among the first N candidates.
    """

    selected_accuracy = []

    best_index = 0
    best_score = scores[0]

    for i in range(len(scores)):

        if scores[i] > best_score:
            best_score = scores[i]
            best_index = i

        selected_accuracy.append(
            int(
                bool(
                    accuracy[best_index]
                )
            )
        )

    return np.asarray(
        selected_accuracy,
        dtype=np.int64,
    )


def compute_summary(
        accuracy,
):
    peak_index = int(
        np.argmax(accuracy)
    )

    peak_n = peak_index + 1

    peak_accuracy = float(
        accuracy[peak_index]
    )

    final_n = len(accuracy)

    final_accuracy = float(
        accuracy[-1]
    )

    degradation = (
        peak_accuracy
        - final_accuracy
    )

    return {
        "peak_n": peak_n,
        "peak_accuracy": peak_accuracy,
        "final_n": final_n,
        "final_accuracy": final_accuracy,
        "degradation": degradation,
    }


def build_method_scores(
        method,
        caution_item,
        neuboots_item=None,
        neuboots_weight=0.8,
):
    details = caution_item[
        "all_detailed_scores"
    ]

    raw_reward = np.asarray(
        [
            item["reward_score"]
            for item in details
        ],
        dtype=float,
    )

    if method == "rm":
        return raw_reward

    if method == "pessimism":
        return np.asarray(
            [
                item["rnd_score"]
                for item in details
            ],
            dtype=float,
        )

    if method == "caution":
        return np.asarray(
            [
                item["combined_score"]
                for item in details
            ],
            dtype=float,
        )

    if method == "neuboots":

        if neuboots_item is None:
            raise ValueError(
                "NeuBoots method requires "
                "--neuboots-results"
            )

        uncertainty = np.asarray(
            neuboots_item[
                "all_neuboots_uncertainty"
            ],
            dtype=float,
        )

        if len(raw_reward) != len(
            uncertainty
        ):
            raise ValueError(
                "Reward / uncertainty "
                "length mismatch"
            )

        return (
            raw_reward
            - neuboots_weight
            * uncertainty
        )

    raise ValueError(
        f"Unsupported method: {method}"
    )


def evaluate(
        caution_data,
        methods,
        neuboots_data=None,
        neuboots_weight=0.8,
):
    num_candidates = None

    correct_counts = {
        method: None
        for method in methods
    }

    for (
        instance_id,
        caution_item,
    ) in caution_data.items():

        accuracy = np.asarray(
            caution_item[
                "all_accuracy"
            ],
            dtype=bool,
        )

        if num_candidates is None:
            num_candidates = len(
                accuracy
            )

            for method in methods:
                correct_counts[
                    method
                ] = np.zeros(
                    num_candidates,
                    dtype=np.int64,
                )

        elif len(accuracy) != (
            num_candidates
        ):
            raise ValueError(
                f"{instance_id}: "
                "candidate count mismatch"
            )

        neuboots_item = None

        if method_requires_neuboots(
            methods
        ):
            if instance_id not in (
                neuboots_data
            ):
                raise ValueError(
                    f"Missing NeuBoots result "
                    f"for {instance_id}"
                )

            neuboots_item = (
                neuboots_data[
                    instance_id
                ]
            )

        for method in methods:

            scores = (
                build_method_scores(
                    method=method,
                    caution_item=(
                        caution_item
                    ),
                    neuboots_item=(
                        neuboots_item
                    ),
                    neuboots_weight=(
                        neuboots_weight
                    ),
                )
            )

            selected = (
                running_selection(
                    scores,
                    accuracy,
                )
            )

            correct_counts[
                method
            ] += selected

    num_problems = len(
        caution_data
    )

    accuracy_by_method = {}

    for method in methods:
        accuracy_by_method[
            method
        ] = (
            correct_counts[method]
            / num_problems
        )

    return (
        accuracy_by_method,
        num_problems,
        num_candidates,
    )


def method_requires_neuboots(
        methods,
):
    return (
        "neuboots"
        in methods
    )


def save_curves_csv(
        path,
        dataset,
        accuracy_by_method,
        neuboots_weight,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "dataset",
                "method",
                "n",
                "accuracy",
                "method_weight",
            ]
        )

        for (
            method,
            accuracy,
        ) in accuracy_by_method.items():

            if method == "neuboots":
                method_weight = (
                    neuboots_weight
                )
            else:
                method_weight = ""

            for index, value in enumerate(
                accuracy
            ):
                writer.writerow(
                    [
                        dataset,
                        method,
                        index + 1,
                        float(value),
                        method_weight,
                    ]
                )


def save_summary_csv(
        path,
        dataset,
        summaries,
        num_problems,
        num_candidates,
        neuboots_weight,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "dataset",
                "method",
                "method_weight",
                "num_problems",
                "num_candidates",
                "peak_n",
                "peak_accuracy",
                "final_n",
                "final_accuracy",
                "degradation",
            ]
        )

        for method, summary in (
            summaries.items()
        ):

            if method == "neuboots":
                method_weight = (
                    neuboots_weight
                )
            else:
                method_weight = ""

            writer.writerow(
                [
                    dataset,
                    method,
                    method_weight,
                    num_problems,
                    num_candidates,
                    summary[
                        "peak_n"
                    ],
                    summary[
                        "peak_accuracy"
                    ],
                    summary[
                        "final_n"
                    ],
                    summary[
                        "final_accuracy"
                    ],
                    summary[
                        "degradation"
                    ],
                ]
            )


def print_summary(
        dataset,
        summaries,
):
    print()
    print("=" * 78)
    print(
        f"Figure 3 evaluation: "
        f"{dataset}"
    )
    print("=" * 78)

    print(
        f"{'Method':<20}"
        f"{'Peak':>12}"
        f"{'Peak N':>10}"
        f"{'Final':>12}"
        f"{'Degradation':>16}"
    )

    print("-" * 78)

    for method, summary in (
        summaries.items()
    ):

        print(
            f"{method:<20}"
            f"{summary['peak_accuracy']:>12.3f}"
            f"{summary['peak_n']:>10d}"
            f"{summary['final_accuracy']:>12.3f}"
            f"{summary['degradation']:>16.3f}"
        )

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--caution-results",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--neuboots-results",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=[
            "rm",
            "pessimism",
            "caution",
        ],
    )

    parser.add_argument(
        "--neuboots-weight",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if (
        "neuboots" in args.methods
        and args.neuboots_results is None
    ):
        raise ValueError(
            "--neuboots-results is required "
            "when using method 'neuboots'"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    caution_data = (
        load_jsonl_by_id(
            args.caution_results
        )
    )

    neuboots_data = None

    if args.neuboots_results:
        neuboots_data = (
            load_jsonl_by_id(
                args.neuboots_results
            )
        )

    if (
        neuboots_data is not None
        and "neuboots" in args.methods
    ):
        caution_ids = set(
            caution_data
        )

        neuboots_ids = set(
            neuboots_data
        )

        if caution_ids != neuboots_ids:
            raise ValueError(
                "Caution / NeuBoots "
                "instance IDs do not match"
            )

    (
        accuracy_by_method,
        num_problems,
        num_candidates,
    ) = evaluate(
        caution_data=caution_data,
        methods=args.methods,
        neuboots_data=(
            neuboots_data
        ),
        neuboots_weight=(
            args.neuboots_weight
        ),
    )

    summaries = {
        method: compute_summary(
            accuracy
        )
        for method, accuracy
        in accuracy_by_method.items()
    }

    result = {
        "dataset": args.dataset,
        "methods": args.methods,
        "num_problems": (
            num_problems
        ),
        "num_candidates": (
            num_candidates
        ),
        "neuboots_weight": (
            args.neuboots_weight
        ),
        "summary": summaries,
        "accuracy_by_n": {
            method: {
                str(n + 1):
                    float(accuracy[n])
                for n in range(
                    len(accuracy)
                )
            }
            for method, accuracy
            in accuracy_by_method.items()
        },
    }

    json_path = os.path.join(
        args.output_dir,
        "results.json",
    )

    curves_path = os.path.join(
        args.output_dir,
        "curves.csv",
    )

    summary_path = os.path.join(
        args.output_dir,
        "summary.csv",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    save_curves_csv(
        path=curves_path,
        dataset=args.dataset,
        accuracy_by_method=(
            accuracy_by_method
        ),
        neuboots_weight=(
            args.neuboots_weight
        ),
    )

    save_summary_csv(
        path=summary_path,
        dataset=args.dataset,
        summaries=summaries,
        num_problems=num_problems,
        num_candidates=num_candidates,
        neuboots_weight=(
            args.neuboots_weight
        ),
    )

    print_summary(
        dataset=args.dataset,
        summaries=summaries,
    )

    print()
    print(
        f"Saved: {json_path}"
    )
    print(
        f"Saved: {curves_path}"
    )
    print(
        f"Saved: {summary_path}"
    )


if __name__ == "__main__":
    main()