"""Read-only Baseline 6 T3 analysis: minimum dummy vs c0 and 2c0.

Inputs:
  - T3 constant_scores.jsonl, first 256 candidate scores
  - Test 1 step_scores.jsonl, same candidate order, SGD minimum step 1
Outputs: candidate-weighted penalty distributions, within-problem paired
correct-vs-incorrect comparisons, original-score Top-K penalty comparisons,
and Best-of-N selection transitions I->C/C->I/C->C/I->I.

No GPU, no model loading, no re-scoring, and no changes to input files.
"""

import argparse
import csv
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import numpy as np

from ensemble.evaluation.baseline6.common import finite, groups


METHODS = ("sgd_min_step1", "c0", "2c0")
ALL_METHODS = ("vanilla",) + METHODS
PENALTY_GROUPS = ("all", "correct", "incorrect", "top16")


def write_csv(path, records):
    if not records:
        raise ValueError(f"No data for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def statistics(values, method, group):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        raise ValueError(f"Empty penalty group: {method}/{group}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"Nonfinite penalty for {method}/{group}")
    return {
        "method": method,
        "group": group,
        "count": int(len(a)),
        "mean_penalty": float(a.mean()),
        "median_penalty": float(np.median(a)),
        "std_penalty": float(a.std()),
        "p10_penalty": float(np.percentile(a, 10)),
        "p90_penalty": float(np.percentile(a, 90)),
        "fraction_reward_decreased": float((a > 1e-9).mean()),
        "fraction_reward_increased": float((a < -1e-9).mean()),
    }


def plot_penalties(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for field, filename, title in (
        ("mean_penalty", "mean_penalty_by_method", "Mean Penalty by Dummy Target"),
        ("median_penalty", "median_penalty_by_method", "Median Penalty by Dummy Target"),
    ):
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        x = np.arange(len(METHODS), dtype=float)
        width = 0.24
        for k, (group, label) in enumerate((
            ("all", "All candidates"),
            ("correct", "Correct candidates"),
            ("incorrect", "Incorrect candidates"),
        )):
            points = [next(r[field] for r in rows if r["method"] == m and r["group"] == group)
                      for m in METHODS]
            ax.bar(x + (k - 1) * width, points, width, label=label)
        ax.axhline(0, color="gray", alpha=0.55, linewidth=0.8)
        ax.set_xticks(x, METHODS)
        ax.set_ylabel("Reward penalty (original - adapted)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=200)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for field, label, style in (
        ("mean_penalty", "Mean", "o-"),
        ("median_penalty", "Median", "s--"),
    ):
        points = [next(r[field] for r in rows if r["method"] == m and r["group"] == "all")
                  for m in METHODS]
        ax.plot(METHODS, points, style, label=label)
    ax.set_ylabel("Reward penalty (original - adapted)")
    ax.set_title("Mean / Median Reward Penalty")
    ax.axhline(0, color="gray", alpha=0.5, linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mean_vs_median_penalty.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    means = [next(r["mean_penalty"] for r in rows
                  if r["method"] == m and r["group"] == "top16")
             for m in METHODS]
    medians = [next(r["median_penalty"] for r in rows
                    if r["method"] == m and r["group"] == "top16")
               for m in METHODS]
    x = np.arange(len(METHODS))
    ax.bar(x - 0.18, means, 0.36, label="Mean")
    ax.bar(x + 0.18, medians, 0.36, label="Median")
    ax.set_xticks(x, METHODS)
    ax.set_ylabel("Reward penalty (original - adapted)")
    ax.set_title("Original-RM Top-16 Penalty: Calibration Check")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "top16_penalty_comparison.png", dpi=200)
    plt.close(fig)


def plot_paired(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(METHODS))
    means = [next(r["mean_correct_minus_incorrect"] for r in rows
                  if r["method"] == m) for m in METHODS]
    medians = [next(r["median_correct_minus_incorrect"] for r in rows
                    if r["method"] == m) for m in METHODS]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - 0.18, means, 0.36, label="Mean of per-problem differences")
    ax.bar(x + 0.18, medians, 0.36, label="Median of per-problem differences")
    ax.axhline(0, linestyle=":", color="gray")
    ax.set_xticks(x, METHODS)
    ax.set_ylabel("Correct mean penalty - Incorrect mean penalty")
    ax.set_title("Within-Problem Paired Penalty Difference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "within_problem_correct_minus_incorrect.png", dpi=200)
    fig.savefig(output / "within_problem_correct_minus_incorrect.pdf")
    plt.close(fig)


def plot_flips(rows, output, nmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = [next(r for r in rows if r["method"] == m and r["N"] == nmax)
             for m in METHODS]
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - 0.18, [r["I_to_C"] for r in final], 0.36,
           label="Incorrect -> Correct")
    ax.bar(x + 0.18, [r["C_to_I"] for r in final], 0.36,
           label="Correct -> Incorrect")
    ax.set_xticks(x, METHODS)
    ax.set_ylabel("Problems")
    ax.set_title(f"Correctness Flips vs Vanilla RM (N={nmax})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "correctness_flips_n256.png", dpi=200)
    fig.savefig(output / "correctness_flips_n256.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        ax.plot([r["N"] for r in subset], [r["flip_rate_percent"] for r in subset],
                marker="o", label=method)
    ns = sorted({r["N"] for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Selection budget N")
    ax.set_ylabel("Flip rate vs Vanilla RM (%)")
    ax.set_title("T3 Selection Flip vs N")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "flip_rate_vs_n.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        ax.plot([r["N"] for r in subset], [r["I_to_C"] for r in subset],
                marker="o", label=f"{method} I->C")
        ax.plot([r["N"] for r in subset], [r["C_to_I"] for r in subset],
                marker="x", linestyle="--", label=f"{method} C->I")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Selection budget N")
    ax.set_ylabel("Number of problems")
    ax.set_title("T3 Correctness Flips vs N")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "correctness_flips_vs_n.png", dpi=200)
    plt.close(fig)


def verify_existing_curves(calculated, path):
    if path is None:
        return
    reference = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            reference[(record["method"], int(record["N"]))] = record
    for row in calculated:
        key = (row["method"], row["N"])
        if key not in reference:
            raise ValueError(f"Missing reference evaluation curve: {key}")
        prior = reference[key]
        for field in ("accuracy_percent", "flip_rate_percent"):
            if abs(float(prior[field]) - row[field]) > 1e-7:
                raise ValueError(
                    f"Existing T3 evaluation mismatch: {key} {field}; "
                    f"old={prior[field]} calculated={row[field]}"
                )
    print("All accuracy/flip rates match existing T3 evaluation curves.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="T3 constant_scores.jsonl")
    parser.add_argument("--sgd-reference", type=Path, required=True,
                        help="Test 1 step_scores.jsonl for same 500x256 candidates")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-problems", type=int, default=500)
    parser.add_argument("--scored-candidates", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--verify-curves", type=Path,
                        help="T3 evaluation/curves.csv")
    args = parser.parse_args()

    if (args.expected_problems <= 0 or args.scored_candidates <= 0 or
            args.scored_candidates & (args.scored_candidates - 1) or
            not (1 <= args.top_k <= args.scored_candidates)):
        parser.error("Invalid problem count, candidate count or Top-K")

    nmax = args.scored_candidates
    ns = [2 ** j for j in range(nmax.bit_length()) if 2 ** j <= nmax]
    penalties = {(method, group): [] for method in METHODS
                 for group in PENALTY_GROUPS}
    paired = {method: [] for method in METHODS}
    per_problem = []
    selection_counts = {(method, n): Counter() for method in ALL_METHODS for n in ns}
    details = []
    problem_count = 0

    source_pairs = zip_longest(groups(args.input, nmax),
                               groups(args.sgd_reference, nmax))
    for own, sgd in source_pairs:
        if own is None or sgd is None:
            raise ValueError("T3 and SGD reference have different problem counts")
        problem_count += 1
        if problem_count > args.expected_problems:
            raise ValueError("More problems than --expected-problems")

        own_ids = [r["candidate_id"] for r in own]
        ref_ids = [r["candidate_id"] for r in sgd]
        if own_ids != ref_ids:
            raise ValueError(f"Candidate order mismatch in problem {problem_count}")
        if any(row.get("finite") is not True for row in own):
            raise ValueError(f"Nonfinite T3 result in problem {problem_count}")
        truth = np.asarray([int(row["correct"]) for row in own], dtype=np.int8)
        if any(type(row["correct"]) not in (int, bool) or row["correct"] not in (0, 1)
               for row in own):
            raise ValueError(f"Invalid correctness in problem {problem_count}")
        if any(int(row["correct"]) != int(truth[j]) for j, row in enumerate(sgd)):
            raise ValueError(f"Correctness disagrees with SGD in problem {problem_count}")
        original = np.asarray([finite(row["reward_model_score"]) for row in own],
                              dtype=np.float64)
        before = np.asarray([finite(row["reward_before"]) for row in own],
                            dtype=np.float64)
        if not np.allclose(original, before, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Original/Before score mismatch in problem {problem_count}")

        scored = {
            "vanilla": original,
            "sgd_min_step1": np.asarray(
                [finite(row["scores_by_step"]["1"]) for row in sgd],
                dtype=np.float64,
            ),
            "c0": np.asarray(
                [finite(row["scores_by_constant"]["c0"]) for row in own],
                dtype=np.float64,
            ),
            "2c0": np.asarray(
                [finite(row["scores_by_constant"]["2c0"]) for row in own],
                dtype=np.float64,
            ),
        }
        top_indices = np.argsort(-original, kind="stable")[:args.top_k]
        for method in METHODS:
            penalty = original - scored[method]
            penalties[(method, "all")].extend(penalty.tolist())
            penalties[(method, "correct")].extend(penalty[truth == 1].tolist())
            penalties[(method, "incorrect")].extend(penalty[truth == 0].tolist())
            penalties[(method, "top16")].extend(penalty[top_indices].tolist())
            correct_pen = penalty[truth == 1]
            incorrect_pen = penalty[truth == 0]
            if correct_pen.size and incorrect_pen.size:
                diff = float(correct_pen.mean() - incorrect_pen.mean())
                paired[method].append(diff)
                per_problem.append({
                    "dataset_index": own[0]["dataset_index"],
                    "instance_id": own[0]["instance_id"],
                    "method": method,
                    "num_correct": int(correct_pen.size),
                    "num_incorrect": int(incorrect_pen.size),
                    "correct_mean_penalty": float(correct_pen.mean()),
                    "incorrect_mean_penalty": float(incorrect_pen.mean()),
                    "difference_correct_minus_incorrect": diff,
                })

        vanilla_indices = {n: int(np.argmax(original[:n])) for n in ns}
        for method in ALL_METHODS:
            for n in ns:
                selected = int(np.argmax(scored[method][:n]))
                baseline = vanilla_indices[n]
                original_correct = int(truth[baseline])
                updated_correct = int(truth[selected])
                counter = selection_counts[(method, n)]
                counter["original_correct"] += original_correct
                counter["updated_correct"] += updated_correct
                if selected == baseline:
                    category = "same"
                elif original_correct == 0 and updated_correct == 1:
                    category = "I_to_C"
                elif original_correct == 1 and updated_correct == 0:
                    category = "C_to_I"
                elif original_correct == 1 and updated_correct == 1:
                    category = "C_to_C"
                else:
                    category = "I_to_I"
                counter[category] += 1
                if method != "vanilla" and n == nmax and selected != baseline:
                    penalty = original - scored[method]
                    details.append({
                        "dataset_index": own[0]["dataset_index"],
                        "instance_id": own[0]["instance_id"],
                        "method": method,
                        "N": n,
                        "category": category,
                        "vanilla_candidate_id": own[baseline]["candidate_id"],
                        "updated_candidate_id": own[selected]["candidate_id"],
                        "vanilla_correct": original_correct,
                        "updated_correct": updated_correct,
                        "vanilla_original_reward": float(original[baseline]),
                        "vanilla_candidate_penalty": float(penalty[baseline]),
                        "updated_candidate_original_reward": float(original[selected]),
                        "updated_candidate_penalty": float(penalty[selected]),
                    })
        if problem_count % 50 == 0:
            print(f"Analyzed {problem_count} problems / "
                  f"{problem_count*nmax:,} candidates", flush=True)

    if problem_count != args.expected_problems:
        raise ValueError(f"Expected {args.expected_problems} problems; got {problem_count}")

    penalty_summary = [statistics(penalties[(method, group)], method, group)
                       for method in METHODS for group in PENALTY_GROUPS]

    paired_summary = []
    for method in METHODS:
        array = np.asarray(paired[method], dtype=np.float64)
        if not len(array):
            raise ValueError(f"No mixed-correctness problems for {method}")
        paired_summary.append({
            "method": method,
            "eligible_problems": int(len(array)),
            "mean_correct_minus_incorrect": float(array.mean()),
            "median_correct_minus_incorrect": float(np.median(array)),
            "fraction_correct_penalty_larger": float((array > 0).mean()),
            "fraction_incorrect_penalty_larger": float((array < 0).mean()),
        })

    flip_summary = []
    for method in ALL_METHODS:
        for n in ns:
            c = selection_counts[(method, n)]
            total = sum(c[k] for k in ("same", "I_to_C", "C_to_I", "C_to_C", "I_to_I"))
            if total != problem_count:
                raise ValueError(f"Incorrect selection total for {method} N={n}")
            net = c["I_to_C"] - c["C_to_I"]
            if net != c["updated_correct"] - c["original_correct"]:
                raise ValueError(f"Inconsistent correctness change for {method} N={n}")
            flip_summary.append({
                "method": method,
                "N": n,
                "num_problems": problem_count,
                "original_accuracy_percent": 100 * c["original_correct"] / problem_count,
                "accuracy_percent": 100 * c["updated_correct"] / problem_count,
                "difference_vs_vanilla_pp": 100 * net / problem_count,
                "flip_count": problem_count - c["same"],
                "flip_rate_percent": 100 * (problem_count - c["same"]) / problem_count,
                "I_to_C": c["I_to_C"],
                "C_to_I": c["C_to_I"],
                "C_to_C": c["C_to_C"],
                "I_to_I": c["I_to_I"],
                "unchanged": c["same"],
                "net_correct": net,
            })

    verify_existing_curves(flip_summary, args.verify_curves)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "penalty_summary_by_method.csv", penalty_summary)
    write_csv(out / "per_problem_penalty_by_method.csv", per_problem)
    write_csv(out / "paired_penalty_summary_by_method.csv", paired_summary)
    write_csv(out / "flip_by_method_and_n.csv", flip_summary)
    if details:
        write_csv(out / "selection_flip_details_n256.csv", details)

    plot_penalties(penalty_summary, out)
    plot_paired(paired_summary, out)
    plot_flips(flip_summary, out, nmax)

    print("\n=== Reward Penalty ===")
    for row in penalty_summary:
        if row["group"] in ("all", "top16"):
            print(f"{row['method']:>14} {row['group']:>6}: "
                  f"mean={row['mean_penalty']:+.6f} "
                  f"median={row['median_penalty']:+.6f}")
    print("\n=== Within-Problem: Correct - Incorrect ===")
    for row in paired_summary:
        print(f"{row['method']:>14}: "
              f"mean difference={row['mean_correct_minus_incorrect']:+.6f} "
              f"correct larger={100*row['fraction_correct_penalty_larger']:.2f}%")
    print(f"\n=== N={nmax}: Selection Correctness Flips ===")
    print("Method         Flip(%)  I->C  C->I  C->C  I->I  Net")
    for row in flip_summary:
        if row["N"] == nmax and row["method"] != "vanilla":
            print(f"{row['method']:>14} {row['flip_rate_percent']:>7.2f} "
                  f"{row['I_to_C']:>5} {row['C_to_I']:>5} "
                  f"{row['C_to_C']:>5} {row['I_to_I']:>5} "
                  f"{row['net_correct']:>+4}")
    print(f"\nAnalysis saved to {out}", flush=True)


if __name__ == "__main__":
    main()
