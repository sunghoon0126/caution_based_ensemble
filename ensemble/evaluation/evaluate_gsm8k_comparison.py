import os
import json
import csv
import argparse

import numpy as np


def load_jsonl_by_id(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
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


def update_running_selection(
    scores,
    accuracy,
):
    """
    For N = 1, ..., num_candidates,
    return whether the best-scoring candidate among
    the first N candidates is correct.
    """

    selected_accuracy = []

    best_index = 0
    best_score = scores[0]

    selected_accuracy.append(
        bool(accuracy[best_index])
    )

    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_index = i

        selected_accuracy.append(
            bool(accuracy[best_index])
        )

    return selected_accuracy


def compute_summary(accuracy_by_n):
    values = np.array(
        [
            accuracy_by_n[str(n)]
            for n in range(
                1,
                len(accuracy_by_n) + 1
            )
        ],
        dtype=float,
    )

    peak_idx = int(values.argmax())
    peak_n = peak_idx + 1
    peak_accuracy = float(values[peak_idx])

    final_accuracy = float(values[-1])

    degradation = (
        peak_accuracy - final_accuracy
    )

    return {
        "peak_n": peak_n,
        "peak_accuracy": peak_accuracy,
        "final_n": len(values),
        "final_accuracy": final_accuracy,
        "degradation": degradation,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--caution-results",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--neuboots-results",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--neuboots-weight",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    caution_data = load_jsonl_by_id(
        args.caution_results
    )

    neuboots_data = load_jsonl_by_id(
        args.neuboots_results
    )

    caution_ids = set(caution_data)
    neuboots_ids = set(neuboots_data)

    if caution_ids != neuboots_ids:
        missing_nb = caution_ids - neuboots_ids
        extra_nb = neuboots_ids - caution_ids

        raise ValueError(
            "Instance IDs do not match.\n"
            f"Missing NeuBoots IDs: {len(missing_nb)}\n"
            f"Extra NeuBoots IDs: {len(extra_nb)}"
        )

    print(
        f"Matched {len(caution_ids)} problems"
    )

    num_candidates = None

    rm_correct_by_n = None
    caution_correct_by_n = None
    neuboots_correct_by_n = None

    for instance_id, caution_item in caution_data.items():
        nb_item = neuboots_data[instance_id]

        details = caution_item[
            "all_detailed_scores"
        ]

        accuracy = np.asarray(
            caution_item["all_accuracy"],
            dtype=bool,
        )

        raw_rewards = np.asarray(
            [
                detail["reward_score"]
                for detail in details
            ],
            dtype=float,
        )

        caution_scores = np.asarray(
            [
                detail["combined_score"]
                for detail in details
            ],
            dtype=float,
        )

        nb_uncertainty = np.asarray(
            nb_item["all_neuboots_uncertainty"],
            dtype=float,
        )

        if not (
            len(raw_rewards)
            == len(caution_scores)
            == len(accuracy)
            == len(nb_uncertainty)
        ):
            raise ValueError(
                f"{instance_id}: candidate length mismatch"
            )

        if num_candidates is None:
            num_candidates = len(
                raw_rewards
            )

            rm_correct_by_n = np.zeros(
                num_candidates,
                dtype=np.int64,
            )

            caution_correct_by_n = np.zeros(
                num_candidates,
                dtype=np.int64,
            )

            neuboots_correct_by_n = np.zeros(
                num_candidates,
                dtype=np.int64,
            )

        elif len(raw_rewards) != num_candidates:
            raise ValueError(
                f"{instance_id}: expected "
                f"{num_candidates} candidates, "
                f"got {len(raw_rewards)}"
            )

        neuboots_scores = (
            raw_rewards
            - args.neuboots_weight
            * nb_uncertainty
        )

        rm_selected = update_running_selection(
            raw_rewards,
            accuracy,
        )

        caution_selected = update_running_selection(
            caution_scores,
            accuracy,
        )

        neuboots_selected = update_running_selection(
            neuboots_scores,
            accuracy,
        )

        rm_correct_by_n += np.asarray(
            rm_selected,
            dtype=np.int64,
        )

        caution_correct_by_n += np.asarray(
            caution_selected,
            dtype=np.int64,
        )

        neuboots_correct_by_n += np.asarray(
            neuboots_selected,
            dtype=np.int64,
        )

    num_problems = len(caution_data)

    rm_accuracy = (
        rm_correct_by_n / num_problems
    )

    caution_accuracy = (
        caution_correct_by_n / num_problems
    )

    neuboots_accuracy = (
        neuboots_correct_by_n / num_problems
    )

    accuracy_by_n = {
        "rm": {
            str(n + 1): float(rm_accuracy[n])
            for n in range(num_candidates)
        },
        "caution": {
            str(n + 1): float(
                caution_accuracy[n]
            )
            for n in range(num_candidates)
        },
        "neuboots": {
            str(n + 1): float(
                neuboots_accuracy[n]
            )
            for n in range(num_candidates)
        },
    }

    summary = {
        "num_problems": num_problems,
        "num_candidates": num_candidates,
        "neuboots_weight":
            args.neuboots_weight,
        "methods": {
            "rm": compute_summary(
                accuracy_by_n["rm"]
            ),
            "caution": compute_summary(
                accuracy_by_n["caution"]
            ),
            "neuboots": compute_summary(
                accuracy_by_n["neuboots"]
            ),
        },
    }

    result = {
        "summary": summary,
        "accuracy_by_n": accuracy_by_n,
    }

    json_path = os.path.join(
        args.output_dir,
        "comparison.json",
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

    csv_path = os.path.join(
        args.output_dir,
        "accuracy_by_n.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "N",
                "RM",
                "Caution",
                "NeuBoots",
            ]
        )

        for n in range(
            1,
            num_candidates + 1,
        ):
            writer.writerow(
                [
                    n,
                    accuracy_by_n[
                        "rm"
                    ][str(n)],
                    accuracy_by_n[
                        "caution"
                    ][str(n)],
                    accuracy_by_n[
                        "neuboots"
                    ][str(n)],
                ]
            )

    print()
    print("=" * 70)
    print("GSM8K comparison")
    print("=" * 70)

    for method in [
        "rm",
        "caution",
        "neuboots",
    ]:
        s = summary["methods"][method]

        print(
            f"{method:10s} | "
            f"peak={s['peak_accuracy']:.3f} "
            f"@ N={s['peak_n']:3d} | "
            f"final={s['final_accuracy']:.3f} | "
            f"degradation={s['degradation']:.3f}"
        )

    print()
    print("Selected N values")
    print("-" * 70)

    selected_n = [
        1, 2, 4, 8, 16, 32,
        64, 128, 256, 512,
    ]

    print(
        f"{'N':>5s} "
        f"{'RM':>10s} "
        f"{'Caution':>10s} "
        f"{'NeuBoots':>10s}"
    )

    for n in selected_n:
        if n > num_candidates:
            continue

        print(
            f"{n:5d} "
            f"{accuracy_by_n['rm'][str(n)]:10.3f} "
            f"{accuracy_by_n['caution'][str(n)]:10.3f} "
            f"{accuracy_by_n['neuboots'][str(n)]:10.3f}"
        )

    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()