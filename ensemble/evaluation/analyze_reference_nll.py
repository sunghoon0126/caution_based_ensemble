
"""Five-method Best-of-N and response-length analysis."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "rm",
    "rm_ref_seq",
    "rm_ref_avg",
    "seq_only",
    "avg_only",
)

LABELS = {
    "rm": "RM only",
    "rm_ref_seq": "RM + Sequence NLL",
    "rm_ref_avg": "RM + Average NLL",
    "seq_only": "Sequence NLL only",
    "avg_only": "Average NLL only",
}


def load_stats(path):
    if path is None:
        return None

    with open(path, encoding="utf-8") as f:
        stats = json.load(f)

    for key in ("reward", "sequence_nll", "average_nll"):
        if key not in stats:
            raise ValueError(f"Missing normalization stats: {key}")

        mean = float(stats[key]["mean"])
        std = float(stats[key]["std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization stats: {key}")

    return stats


def read_groups(paths, expected_candidates):
    """Yield candidates grouped in original problem/sample order."""
    current_key = None
    current_group = []
    previous_dataset_index = -1
    experiment = None

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
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
                elif key[:3] != experiment:
                    raise ValueError("Mixed datasets, splits, or trials")

                if key != current_key:
                    if current_group:
                        if len(current_group) != expected_candidates:
                            raise ValueError(
                                f"{current_key}: expected "
                                f"{expected_candidates}, got "
                                f"{len(current_group)}"
                            )
                        yield current_group

                    if key[3] <= previous_dataset_index:
                        raise ValueError(
                            "Problems not in increasing dataset_index order"
                        )

                    previous_dataset_index = key[3]
                    current_key = key
                    current_group = []

                sample_index = int(row["sample_index"])
                if sample_index != len(current_group):
                    raise ValueError(
                        f"{path}:{line_no}: unexpected sample_index "
                        f"{sample_index}; expected {len(current_group)}"
                    )

                reward = float(row["reward_model_score"])
                seq = float(row["sequence_nll"])
                avg = float(row["average_nll"])
                length = int(row["token_count"])

                if not all(map(math.isfinite, (reward, seq, avg))):
                    raise ValueError(f"Non-finite score at {key}")

                if length <= 0:
                    raise ValueError(f"Invalid token_count at {key}")

                correct = row["correct"]
                if not (
                    isinstance(correct, (bool, int, float))
                    and correct in (0, 1, False, True)
                ):
                    raise ValueError(f"Invalid correct value at {key}")

                current_group.append(
                    (reward, seq, avg, length, int(correct))
                )

    if current_group:
        if len(current_group) != expected_candidates:
            raise ValueError(
                f"{current_key}: incomplete candidate group"
            )
        yield current_group


def make_scores(candidate, seq_lambda, avg_lambda, stats):
    reward, seq, avg, _, _ = candidate

    if stats is not None:
        reward = (
            reward - stats["reward"]["mean"]
        ) / stats["reward"]["std"]

        seq = (
            seq - stats["sequence_nll"]["mean"]
        ) / stats["sequence_nll"]["std"]

        avg = (
            avg - stats["average_nll"]["mean"]
        ) / stats["average_nll"]["std"]

    return {
        "rm": reward,
        "rm_ref_seq": reward - seq_lambda * seq,
        "rm_ref_avg": reward - avg_lambda * avg,
        "seq_only": -seq,
        "avg_only": -avg,
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_line_plot(path, x_values, series, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHODS:
        ax.plot(
            x_values,
            series[method],
            marker="o",
            linewidth=2,
            markersize=5,
            label=LABELS[method],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(n) for n in x_values])
    ax.set_xlabel("Number of candidates (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_length_distribution(path, length_counts, bin_width=20):
    lengths = np.array(sorted(length_counts), dtype=np.int64)
    frequencies = np.array(
        [length_counts[t] for t in lengths], dtype=np.int64
    )

    bins = np.arange(
        0,
        lengths.max() + 2 * bin_width,
        bin_width,
    )

    hist, edges = np.histogram(
        lengths,
        bins=bins,
        weights=frequencies,
    )

    expanded = np.repeat(lengths, frequencies)
    median = float(np.median(expanded))
    p90 = float(np.percentile(expanded, 90))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        edges[:-1],
        hist,
        width=np.diff(edges),
        align="edge",
        alpha=0.75,
        edgecolor="none",
    )

    ax.axvline(
        median,
        linestyle="--",
        label=f"Median: {median:.0f} tokens",
    )
    ax.axvline(
        p90,
        linestyle=":",
        label=f"P90: {p90:.0f} tokens",
    )

    ax.set_xlabel("Generated response length (tokens)")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Overall generated response length distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)

    return {
        "mean": float(expanded.mean()),
        "median": median,
        "p90": p90,
        "p95": float(np.percentile(expanded, 95)),
        "p99": float(np.percentile(expanded, 99)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "count": int(frequencies.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-paths",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--seq-lambda",
        type=float,
        default=0.1,
        help="Raw-score sequence lambda from previous sweep",
    )
    parser.add_argument(
        "--avg-lambda",
        type=float,
        default=10.0,
        help="Raw-score average lambda from previous sweep",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=None,
        help="Independent calibration stats (optional)",
    )
    args = parser.parse_args()

    if args.expected_candidates <= 0:
        raise ValueError("expected-candidates must be positive")

    for lam in (args.seq_lambda, args.avg_lambda):
        if not math.isfinite(lam) or lam < 0:
            raise ValueError("Lambda must be finite and nonnegative")

    stats = load_stats(args.normalization_stats)

    seq_lambda = args.seq_lambda
    avg_lambda = args.avg_lambda

    # Convert raw lambdas so normalization preserves the
    # original candidate rankings.
    if stats is not None:
        reward_std = stats["reward"]["std"]
        seq_lambda *= (
            stats["sequence_nll"]["std"] / reward_std
        )
        avg_lambda *= (
            stats["average_nll"]["std"] / reward_std
        )

    print(f"Mode: {'normalized' if stats else 'raw'}")
    print(f"Effective sequence lambda: {seq_lambda:.8g}")
    print(f"Effective average lambda: {avg_lambda:.8g}")

    n_values = []
    n = 1
    while n <= args.expected_candidates:
        n_values.append(n)
        n *= 2

    if n_values[-1] != args.expected_candidates:
        raise ValueError(
            "expected-candidates must be a power of 2"
        )

    n_set = set(n_values)
    accuracy_sums = defaultdict(int)
    length_sums = defaultdict(int)
    length_counts = Counter()
    num_problems = 0

    for group in read_groups(
        args.input_paths, args.expected_candidates
    ):
        num_problems += 1

        best_scores = {m: -math.inf for m in METHODS}
        best_correct = {m: 0 for m in METHODS}
        best_length = {m: 0 for m in METHODS}

        for index, candidate in enumerate(group, start=1):
            _, _, _, length, correct = candidate
            length_counts[length] += 1

            scores = make_scores(
                candidate, seq_lambda, avg_lambda, stats
            )

            for method in METHODS:
                # Strict > keeps earliest candidate when tied.
                if scores[method] > best_scores[method]:
                    best_scores[method] = scores[method]
                    best_correct[method] = correct
                    best_length[method] = length

            if index in n_set:
                for method in METHODS:
                    accuracy_sums[(method, index)] += (
                        best_correct[method]
                    )
                    length_sums[(method, index)] += (
                        best_length[method]
                    )

        if num_problems % 100 == 0:
            print(
                f"Processed {num_problems} problems",
                flush=True,
            )

    if num_problems == 0:
        raise RuntimeError("No candidates found")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    accuracy_curves = {}
    length_curves = {}
    accuracy_rows = []
    length_rows = []
    summary_rows = []

    for method in METHODS:
        accuracies = []
        lengths = []

        for n in n_values:
            acc = 100 * accuracy_sums[
                (method, n)
            ] / num_problems

            mean_length = (
                length_sums[(method, n)] / num_problems
            )

            accuracies.append(acc)
            lengths.append(mean_length)

            accuracy_rows.append({
                "method": method,
                "N": n,
                "accuracy_percent": acc,
            })
            length_rows.append({
                "method": method,
                "N": n,
                "mean_selected_token_count": mean_length,
            })

        accuracy_curves[method] = accuracies
        length_curves[method] = lengths

        peak_idx = int(np.argmax(accuracies))
        summary_rows.append({
            "method": method,
            "peak_n": n_values[peak_idx],
            "peak_accuracy_percent": accuracies[peak_idx],
            "final_accuracy_percent": accuracies[-1],
            "degradation_pp": (
                accuracies[peak_idx] - accuracies[-1]
            ),
            "mean_length_at_512": lengths[-1],
        })

    write_csv(
        output / "accuracy_curves.csv",
        ["method", "N", "accuracy_percent"],
        accuracy_rows,
    )

    write_csv(
        output / "selected_length_curves.csv",
        ["method", "N", "mean_selected_token_count"],
        length_rows,
    )

    write_csv(
        output / "summary.csv",
        [
            "method",
            "peak_n",
            "peak_accuracy_percent",
            "final_accuracy_percent",
            "degradation_pp",
            "mean_length_at_512",
        ],
        summary_rows,
    )

    total = sum(length_counts.values())
    cumulative = 0
    distribution_rows = []

    for length, count in sorted(length_counts.items()):
        cumulative += count
        distribution_rows.append({
            "token_count": length,
            "count": count,
            "fraction": count / total,
            "cumulative_fraction": cumulative / total,
        })

    write_csv(
        output / "length_distribution.csv",
        [
            "token_count",
            "count",
            "fraction",
            "cumulative_fraction",
        ],
        distribution_rows,
    )

    save_line_plot(
        output / "accuracy_five_methods.png",
        n_values,
        accuracy_curves,
        "Accuracy (%)",
        "Best-of-N Accuracy: Five Methods",
    )

    save_line_plot(
        output / "selected_response_length.png",
        n_values,
        length_curves,
        "Mean selected response length (tokens)",
        "Selected Response Length vs N",
    )

    length_stats = plot_length_distribution(
        output / "overall_response_length_distribution.png",
        length_counts,
    )

    metadata = {
        "num_problems": num_problems,
        "num_candidates": total,
        "raw_seq_lambda": args.seq_lambda,
        "raw_avg_lambda": args.avg_lambda,
        "effective_seq_lambda": seq_lambda,
        "effective_avg_lambda": avg_lambda,
        "normalization_stats_file": (
            str(args.normalization_stats) if stats else None
        ),
        "overall_length_stats": length_stats,
    }

    with open(
        output / "analysis_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nEvaluation summary")
    for row in summary_rows:
        print(
            f"{row['method']:12s} "
            f"peak={row['peak_accuracy_percent']:6.2f}% "
            f"N={row['peak_n']:3d} "
            f"final={row['final_accuracy_percent']:6.2f}% "
            f"deg={row['degradation_pp']:5.2f}pp "
            f"length@512={row['mean_length_at_512']:7.2f}"
        )

    print("\nLength statistics:", length_stats)
    print("Saved outputs to:", output)


if __name__ == "__main__":
    main()
