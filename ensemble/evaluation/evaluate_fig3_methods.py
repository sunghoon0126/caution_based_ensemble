import os
import csv
import json
import argparse

import numpy as np


SUPPORTED_METHODS = [
    "rm",
    "pessimism",
    "caution",
    "neuboots_pessimism",
    "rm_neuboots",
]


METHOD_DISPLAY_NAMES = {
    "rm": "RM",
    "pessimism": "Pessimism",
    "caution": "RM + Pessimism",
    "neuboots_pessimism": "NeuBoots Pessimism",
    "rm_neuboots": "RM + NeuBoots",
}


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
            instance_id = item["instance_id"]

            if instance_id in data:
                raise ValueError(
                    f"Duplicate instance_id: {instance_id}"
                )

            data[instance_id] = item

    return data


def method_requires_neuboots(methods):
    return any(
        method in methods
        for method in [
            "neuboots_pessimism",
            "rm_neuboots",
        ]
    )


def running_selection(
        scores,
        accuracy,
):
    """
    For N = 1, ..., num_candidates:

    Among the first N candidates,
    select the candidate with the highest score
    and record whether that candidate is correct.
    """

    if len(scores) == 0:
        raise ValueError(
            "scores must contain at least one candidate"
        )

    if len(scores) != len(accuracy):
        raise ValueError(
            "scores and accuracy length mismatch"
        )

    selected_accuracy = np.zeros(
        len(scores),
        dtype=np.int64,
    )

    best_index = 0
    best_score = scores[0]

    selected_accuracy[0] = int(
        bool(accuracy[0])
    )

    for i in range(
        1,
        len(scores),
    ):
        if scores[i] > best_score:
            best_score = scores[i]
            best_index = i

        selected_accuracy[i] = int(
            bool(
                accuracy[best_index]
            )
        )

    return selected_accuracy


def compute_summary(accuracy):
    """
    Compute Figure 3 summary statistics.

    Peak:
        maximum accuracy across N.

    Final:
        accuracy at maximum available N.

    Degradation:
        Peak - Final.
    """

    accuracy = np.asarray(
        accuracy,
        dtype=float,
    )

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
    """
    Construct candidate selection scores.

    rm:
        raw RM reward

    pessimism:
        original Caution RND pessimism score.
        Larger score means less uncertainty / more preferred.

    caution:
        original stored RM + Pessimism combined score.

    neuboots_pessimism:
        - NeuBoots uncertainty.
        Since selection uses argmax,
        this selects minimum-uncertainty candidates.

    rm_neuboots:
        RM reward - lambda * NeuBoots uncertainty.
    """

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

    if method in [
        "neuboots_pessimism",
        "rm_neuboots",
    ]:
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
                "Reward / NeuBoots uncertainty "
                "length mismatch"
            )

        if method == "neuboots_pessimism":
            return -uncertainty

        if method == "rm_neuboots":
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
    """
    Evaluate all selected methods over
    N = 1, ..., num_candidates.
    """

    num_candidates = None

    correct_counts = {
        method: None
        for method in methods
    }

    use_neuboots = (
        method_requires_neuboots(
            methods
        )
    )

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
                f"expected {num_candidates} candidates, "
                f"got {len(accuracy)}"
            )

        neuboots_item = None

        if use_neuboots:
            if neuboots_data is None:
                raise ValueError(
                    "NeuBoots data is missing"
                )

            if instance_id not in (
                neuboots_data
            ):
                raise ValueError(
                    f"Missing NeuBoots result "
                    f"for instance_id={instance_id}"
                )

            neuboots_item = (
                neuboots_data[
                    instance_id
                ]
            )

        for method in methods:

            scores = build_method_scores(
                method=method,
                caution_item=caution_item,
                neuboots_item=(
                    neuboots_item
                ),
                neuboots_weight=(
                    neuboots_weight
                ),
            )

            if len(scores) != (
                num_candidates
            ):
                raise ValueError(
                    f"{instance_id}: "
                    f"{method} score length "
                    f"is {len(scores)}, "
                    f"expected {num_candidates}"
                )

            selected = (
                running_selection(
                    scores=scores,
                    accuracy=accuracy,
                )
            )

            correct_counts[
                method
            ] += selected

    num_problems = len(
        caution_data
    )

    if num_problems == 0:
        raise ValueError(
            "No problems found in input data"
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


def get_method_weight(
        method,
        neuboots_weight,
):
    """
    Only RM + NeuBoots explicitly uses
    the CLI NeuBoots lambda.

    Original Caution combined_score is
    already stored in the reference result,
    so its weight is not recomputed here.
    """

    if method == "rm_neuboots":
        return neuboots_weight

    return ""


def save_curves_csv(
        path,
        dataset,
        accuracy_by_method,
        neuboots_weight,
):
    """
    Long-format Figure 3 curve data.

    One row:
        dataset / method / N / accuracy
    """

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
                "method_display_name",
                "method_weight",
                "n",
                "accuracy",
            ]
        )

        for (
            method,
            accuracy,
        ) in accuracy_by_method.items():

            method_weight = (
                get_method_weight(
                    method=method,
                    neuboots_weight=(
                        neuboots_weight
                    ),
                )
            )

            for index, value in enumerate(
                accuracy
            ):
                writer.writerow(
                    [
                        dataset,
                        method,
                        METHOD_DISPLAY_NAMES[
                            method
                        ],
                        method_weight,
                        index + 1,
                        float(value),
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
    """
    Table-oriented summary.

    One row per method.
    """

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
                "method_display_name",
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

        for (
            method,
            summary,
        ) in summaries.items():

            method_weight = (
                get_method_weight(
                    method=method,
                    neuboots_weight=(
                        neuboots_weight
                    ),
                )
            )

            writer.writerow(
                [
                    dataset,
                    method,
                    METHOD_DISPLAY_NAMES[
                        method
                    ],
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
    print(
        "=" * 88
    )

    print(
        f"Figure 3 evaluation: "
        f"{dataset}"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Method':<26}"
        f"{'Peak':>12}"
        f"{'Peak N':>10}"
        f"{'Final':>12}"
        f"{'Degradation':>16}"
    )

    print(
        "-" * 88
    )

    for (
        method,
        summary,
    ) in summaries.items():

        display_name = (
            METHOD_DISPLAY_NAMES[
                method
            ]
        )

        print(
            f"{display_name:<26}"
            f"{summary['peak_accuracy']:>12.3f}"
            f"{summary['peak_n']:>10d}"
            f"{summary['final_accuracy']:>12.3f}"
            f"{summary['degradation']:>16.3f}"
        )

    print(
        "=" * 88
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Dataset name stored in output "
            "metadata, e.g. gsm8k, math500, bbh."
        ),
    )

    parser.add_argument(
        "--caution-results",
        type=str,
        required=True,
        help=(
            "Original Caution results.jsonl."
        ),
    )

    parser.add_argument(
        "--neuboots-results",
        type=str,
        default=None,
        help=(
            "NeuBoots candidate scoring "
            "results.jsonl."
        ),
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
        help=(
            "Methods to evaluate."
        ),
    )

    parser.add_argument(
        "--neuboots-weight",
        type=float,
        default=0.8,
        help=(
            "Lambda used for "
            "RM + NeuBoots."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    use_neuboots = (
        method_requires_neuboots(
            args.methods
        )
    )

    if (
        use_neuboots
        and args.neuboots_results
        is None
    ):
        raise ValueError(
            "--neuboots-results is required "
            "when using "
            "neuboots_pessimism or "
            "rm_neuboots"
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

    if use_neuboots:
        neuboots_data = (
            load_jsonl_by_id(
                args.neuboots_results
            )
        )

        caution_ids = set(
            caution_data
        )

        neuboots_ids = set(
            neuboots_data
        )

        if caution_ids != (
            neuboots_ids
        ):
            missing_neuboots = (
                caution_ids
                - neuboots_ids
            )

            extra_neuboots = (
                neuboots_ids
                - caution_ids
            )

            raise ValueError(
                "Caution / NeuBoots "
                "instance IDs do not match.\n"
                f"Missing NeuBoots IDs: "
                f"{len(missing_neuboots)}\n"
                f"Extra NeuBoots IDs: "
                f"{len(extra_neuboots)}"
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
        for (
            method,
            accuracy,
        ) in accuracy_by_method.items()
    }

    result = {
        "dataset": args.dataset,
        "methods": args.methods,
        "method_display_names": {
            method:
                METHOD_DISPLAY_NAMES[
                    method
                ]
            for method in args.methods
        },
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
                str(index + 1):
                    float(
                        accuracy[index]
                    )
                for index in range(
                    len(accuracy)
                )
            }
            for (
                method,
                accuracy,
            ) in accuracy_by_method.items()
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
        f"Saved JSON : {json_path}"
    )
    print(
        f"Saved curve: {curves_path}"
    )
    print(
        f"Saved table: {summary_path}"
    )


if __name__ == "__main__":
    main()