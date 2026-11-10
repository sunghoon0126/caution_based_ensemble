
"""Best-of-N evaluation for likelihood-based reward pessimism."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_lambdas(value):
    result = [float(v.strip()) for v in value.split(",")]
    if not result or any(
        not math.isfinite(v) or v < 0 for v in result
    ):
        raise ValueError("Lambda values must be finite and nonnegative")
    return list(dict.fromkeys(result))


def load_problem_groups(paths, expected_candidates):
    """Stream problems in original dataset/sample order."""
    current_key = None
    previous_index = None
    experiment = None
    candidates = []

    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue

                row = json.loads(line)

                key = (
                    str(row["dataset"]),
                    str(row["split"]),
                    int(row["trial_id"]),
                    int(row["dataset_index"]),
                    str(row["instance_id"]),
                )

                if experiment is None:
                    experiment = key[:3]
                elif experiment != key[:3]:
                    raise ValueError(
                        f"Mixed dataset/split/trial at {path}:{line_number}"
                    )

                if key != current_key:
                    if candidates:
                        if len(candidates) != expected_candidates:
                            raise ValueError(
                                f"{current_key}: expected "
                                f"{expected_candidates} candidates, "
                                f"got {len(candidates)}"
                            )
                        yield current_key, candidates

                    if (
                        previous_index is not None
                        and key[3] <= previous_index
                    ):
                        raise ValueError(
                            f"Non-increasing dataset_index: {key[3]}"
                        )

                    current_key = key
                    previous_index = key[3]
                    candidates = []

                sample_index = int(row["sample_index"])
                if sample_index != len(candidates):
                    raise ValueError(
                        f"{key}: expected sample_index "
                        f"{len(candidates)}, got {sample_index}"
                    )

                reward = float(row["reward_model_score"])
                seq = float(row["sequence_nll"])
                avg = float(row["average_nll"])

                if not all(map(math.isfinite, (reward, seq, avg))):
                    raise ValueError(f"Non-finite score: {key}")

                correct = row["correct"]
                if not (
                    isinstance(correct, bool)
                    or (
                        isinstance(correct, (int, float))
                        and correct in (0, 1)
                    )
                ):
                    raise ValueError(f"Invalid correctness: {key}")

                candidates.append(
                    (reward, seq, avg, int(correct))
                )

    if candidates:
        if len(candidates) != expected_candidates:
            raise ValueError(
                f"{current_key}: expected {expected_candidates}, "
                f"got {len(candidates)}"
            )
        yield current_key, candidates


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output_dir, curves, methods, n_values):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: skipping PNG plots")
        return

    for family in ("rm_ref_seq", "rm_ref_avg"):
        fig, ax = plt.subplots(figsize=(9, 5))

        for method, lam in methods:
            if method not in ("rm", family):
                continue

            points = [
                row for row in curves
                if row["method"] == method
                and row["lambda"] == lam
            ]

            label = (
                "RM only" if method == "rm"
                else f"{family}, lambda={lam:g}"
            )

            ax.plot(
                [p["N"] for p in points],
                [p["accuracy_percent"] for p in points],
                marker="o",
                label=label,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.set_xlabel("Best-of-N")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"GSM8K Best-of-N: {family}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"curves_{family}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-paths", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    parser.add_argument(
        "--expected-candidates", type=int, default=512
    )
    parser.add_argument(
        "--seq-lambdas",
        default="0.0005,0.001,0.005,0.01,0.02",
    )
    parser.add_argument(
        "--avg-lambdas",
        default="0.05,0.1,0.2,0.5,1.0,2.0",
    )
    args = parser.parse_args()

    if args.expected_candidates <= 0:
        raise ValueError("Expected candidates must be positive")

    seq_lambdas = parse_lambdas(args.seq_lambdas)
    avg_lambdas = parse_lambdas(args.avg_lambdas)

    methods = (
        [("rm", 0.0)]
        + [("rm_ref_seq", x) for x in seq_lambdas]
        + [("rm_ref_avg", x) for x in avg_lambdas]
    )

    n_values = []
    n = 1
    while n <= args.expected_candidates:
        n_values.append(n)
        n *= 2

    if n_values[-1] != args.expected_candidates:
        raise ValueError(
            "expected-candidates must be a power of two"
        )

    n_set = set(n_values)
    correct_counts = defaultdict(int)
    num_problems = 0
    experiment = None

    for key, candidates in load_problem_groups(
        args.input_paths, args.expected_candidates
    ):
        if experiment is None:
            experiment = key[:3]

        num_problems += 1

        for method, lam in methods:
            best_score = -math.inf
            best_correct = 0

            for index, (reward, seq, avg, correct) in enumerate(
                candidates, start=1
            ):
                if method == "rm":
                    score = reward
                elif method == "rm_ref_seq":
                    score = reward - lam * seq
                else:
                    score = reward - lam * avg

                # Strict > preserves earliest-candidate tie-breaking.
                if score > best_score:
                    best_score = score
                    best_correct = correct

                if index in n_set:
                    correct_counts[(method, lam, index)] += best_correct

        if num_problems % 100 == 0:
            print(f"Evaluated {num_problems} problems", flush=True)

    if num_problems == 0:
        raise RuntimeError("No candidates found")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, split, trial_id = experiment
    curves = []
    summaries = []

    for method, lam in methods:
        points = []

        for n in n_values:
            count = correct_counts[(method, lam, n)]
            accuracy = count / num_problems

            row = {
                "dataset": dataset,
                "split": split,
                "trial_id": trial_id,
                "method": method,
                "lambda": lam,
                "N": n,
                "num_problems": num_problems,
                "num_correct": count,
                "accuracy": accuracy,
                "accuracy_percent": 100 * accuracy,
            }
            curves.append(row)
            points.append(row)

        # First maximum wins, corresponding to the smallest peak N.
        peak = max(points, key=lambda row: row["accuracy"])
        final = points[-1]

        summaries.append({
            "dataset": dataset,
            "trial_id": trial_id,
            "method": method,
            "lambda": lam,
            "peak_n": peak["N"],
            "peak_accuracy_percent": peak["accuracy_percent"],
            "final_n": final["N"],
            "final_accuracy_percent": final["accuracy_percent"],
            "degradation_percent": (
                peak["accuracy_percent"]
                - final["accuracy_percent"]
            ),
        })

    save_csv(
        output_dir / "curves.csv",
        curves,
        [
            "dataset", "split", "trial_id", "method",
            "lambda", "N", "num_problems", "num_correct",
            "accuracy", "accuracy_percent",
        ],
    )
    save_csv(
        output_dir / "summary.csv",
        summaries,
        [
            "dataset", "trial_id", "method", "lambda",
            "peak_n", "peak_accuracy_percent", "final_n",
            "final_accuracy_percent", "degradation_percent",
        ],
    )

    plot_curves(output_dir, curves, methods, n_values)

    print(f"Completed {num_problems} problems")
    print(f"Results saved to {output_dir}")
    print()
    for row in summaries:
        print(
            f"{row['method']:12s} "
            f"lambda={row['lambda']:8g} "
            f"peak={row['peak_accuracy_percent']:6.2f}% "
            f"N={row['peak_n']:3d} "
            f"final={row['final_accuracy_percent']:6.2f}% "
            f"deg={row['degradation_percent']:5.2f}pp"
        )


if __name__ == "__main__":
    main()
