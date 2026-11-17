"""Compare correct vs incorrect reward penalties for Baseline 6 T1/T2/T3.

Produces one grouped bar plot for all 9 unique configurations:
  T1: SGD 1,2,4,8 steps
  T2: Adam 1,2,4 steps
  T3: score - c0, score - 2c0 (SGD 1 step)

Penalty = original reward - updated reward, including negative penalties.
Uses the same 500 problems and first 256 candidates in every experiment.
Pooled averages weight individual candidates equally; the correct/incorrect
groups can contain different numbers of candidates.

Reads existing JSONL only, without loading a model or using a GPU.
"""
import argparse
import csv
import math
from itertools import zip_longest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ensemble.evaluation.baseline6.common import finite, groups


CONFIGS = (
    ("sgd_1", "SGD\n1 step", "Test 1: SGD"),
    ("sgd_2", "SGD\n2 steps", "Test 1: SGD"),
    ("sgd_4", "SGD\n4 steps", "Test 1: SGD"),
    ("sgd_8", "SGD\n8 steps", "Test 1: SGD"),
    ("adam_1", "Adam\n1 step", "Test 2: Adam"),
    ("adam_2", "Adam\n2 steps", "Test 2: Adam"),
    ("adam_4", "Adam\n4 steps", "Test 2: Adam"),
    ("const_c0", "Constant\nc0", "Test 3: Constant"),
    ("const_2c0", "Constant\n2c0", "Test 3: Constant"),
)
METHOD_KEYS = [k for k, _, _ in CONFIGS]


def verify_source(row, other, description, problem_idx, candidate_idx):
    if row["candidate_id"] != other["candidate_id"]:
        raise ValueError(
            f"{description} candidate mismatch at problem {problem_idx} "
            f"candidate {candidate_idx}"
        )
    if row["correct"] != other["correct"]:
        raise ValueError(
            f"{description} correctness mismatch at problem {problem_idx} "
            f"candidate {candidate_idx}"
        )
    left = finite(row["reward_model_score"])
    right = finite(other["reward_model_score"])
    if not math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(
            f"{description} original RM score mismatch at problem "
            f"{problem_idx} candidate {candidate_idx}"
        )
    if other.get("finite") is not True:
        raise ValueError(f"Invalid {description} result at problem {problem_idx}")


def summarize(values, method, label, test, group):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"Empty or nonfinite penalty data: {method}/{group}")
    return {
        "test": test,
        "method": method,
        "label": label.replace("\n", " "),
        "correctness": group,
        "n_candidates": int(a.size),
        "mean_penalty": float(np.mean(a)),
        "median_penalty": float(np.median(a)),
        "std_penalty": float(np.std(a)),
        "p10_penalty": float(np.percentile(a, 10)),
        "p90_penalty": float(np.percentile(a, 90)),
        "frac_positive_penalty": float(np.mean(a > 1e-9)),
        "frac_negative_penalty": float(np.mean(a < -1e-9)),
    }


def make_plot(summary, metric, out, dpi):
    by_key = {(r["method"], r["correctness"]): r for r in summary}
    x = np.arange(len(CONFIGS), dtype=float)
    width = 0.37
    fig, ax = plt.subplots(figsize=(16, 6.4))
    correct = [
        by_key[(key, "correct")][metric] for key, _, _ in CONFIGS
    ]
    incorrect = [
        by_key[(key, "incorrect")][metric] for key, _, _ in CONFIGS
    ]
    a = ax.bar(x - width / 2, correct, width, label="Correct")
    b = ax.bar(x + width / 2, incorrect, width, label="Incorrect")
    ax.set_xticks(x, [label for _, label, _ in CONFIGS])
    ax.set_ylabel("Reward penalty = original RM score - updated RM score")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(3.5, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(6.5, color="gray", linestyle="--", alpha=0.4)
    ax.text(1.5, 1.02, "Test 1 · SGD (LR=1e-6)", ha="center",
            transform=ax.get_xaxis_transform())
    ax.text(5.0, 1.02, "Test 2 · Adam (LR=3e-8)", ha="center",
            transform=ax.get_xaxis_transform())
    ax.text(7.5, 1.02, "Test 3 · SGD (LR=1e-6)", ha="center",
            transform=ax.get_xaxis_transform())
    title = "Mean Reward Penalty by Correctness" if metric == "mean_penalty" \
        else "Median Reward Penalty by Correctness"
    ax.set_title(f"Baseline 6 · T1 / T2 / T3 · {title}", pad=37)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    for bars in (a, b):
        for patch in bars:
            height = patch.get_height()
            if not math.isfinite(height):
                continue
            ax.annotate(
                f"{height:+.3f}",
                (patch.get_x() + patch.get_width()/2, height),
                xytext=(0, 3 if height >= 0 else -3),
                textcoords="offset points",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=7.5,
            )
    fig.tight_layout()
    name = ("mean_penalty_correct_incorrect_all_tests"
            if metric == "mean_penalty"
            else "median_penalty_correct_incorrect_all_tests")
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgd", type=Path, required=True,
                        help="T1 SGD step_scores.jsonl")
    parser.add_argument("--adam", type=Path, required=True,
                        help="T2 Adam step_scores.jsonl")
    parser.add_argument("--constant", type=Path, required=True,
                        help="T3 constant_scores.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-problems", type=int, default=500)
    parser.add_argument("--scored-candidates", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.expected_problems <= 0 or args.scored_candidates <= 0 or args.dpi <= 0:
        parser.error("Counts and DPI must be positive")

    penalties = {(key, correctness): []
                 for key in METHOD_KEYS
                 for correctness in ("correct", "incorrect")}
    count_correct = 0
    count_incorrect = 0
    num_problems = 0

    for pair in zip_longest(
        groups(args.sgd, args.scored_candidates),
        groups(args.adam, args.scored_candidates),
        groups(args.constant, args.scored_candidates),
    ):
        t1, t2, t3 = pair
        if any(group is None for group in pair):
            raise ValueError("T1/T2/T3 have different numbers of problems")
        num_problems += 1
        if num_problems > args.expected_problems:
            raise ValueError("More problems than requested")
        for index, (sgd, adam, const) in enumerate(zip(t1, t2, t3)):
            verify_source(sgd, sgd, "T1", num_problems, index)
            verify_source(sgd, adam, "T2", num_problems, index)
            verify_source(sgd, const, "T3", num_problems, index)
            correctness = sgd["correct"]
            if type(correctness) not in (bool, int) or correctness not in (0, 1):
                raise ValueError(f"Invalid correctness at problem {num_problems}/{index}")
            group = "correct" if correctness == 1 else "incorrect"
            if correctness == 1:
                count_correct += 1
            else:
                count_incorrect += 1
            original = finite(sgd["reward_model_score"])
            all_scores = {
                "sgd_1": finite(sgd["scores_by_step"]["1"]),
                "sgd_2": finite(sgd["scores_by_step"]["2"]),
                "sgd_4": finite(sgd["scores_by_step"]["4"]),
                "sgd_8": finite(sgd["scores_by_step"]["8"]),
                "adam_1": finite(adam["scores_by_step"]["1"]),
                "adam_2": finite(adam["scores_by_step"]["2"]),
                "adam_4": finite(adam["scores_by_step"]["4"]),
                "const_c0": finite(const["scores_by_constant"]["c0"]),
                "const_2c0": finite(const["scores_by_constant"]["2c0"]),
            }
            for method, updated in all_scores.items():
                penalties[(method, group)].append(finite(original - updated))
        if num_problems % 50 == 0:
            print(f"Read {num_problems}/{args.expected_problems} problems", flush=True)

    if num_problems != args.expected_problems:
        raise ValueError(f"Expected {args.expected_problems} problems, found {num_problems}")
    if count_correct == 0 or count_incorrect == 0:
        raise ValueError("Need both correct and incorrect candidates")

    summary = []
    for method, label, test in CONFIGS:
        for group in ("correct", "incorrect"):
            summary.append(summarize(penalties[(method, group)],
                                     method, label, test, group))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "penalty_correct_incorrect_all_tests.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    make_plot(summary, "mean_penalty", args.output_dir, args.dpi)
    make_plot(summary, "median_penalty", args.output_dir, args.dpi)

    print("\nReward penalty: original - updated (positive = reduced reward)")
    print(f"Problems={num_problems}, correct candidates={count_correct}, "
          f"incorrect candidates={count_incorrect}")
    print("Method          correct_mean    incorrect_mean    gap(C-I)")
    lookup = {(r["method"], r["correctness"]): r for r in summary}
    for method in METHOD_KEYS:
        c = lookup[(method, "correct")]["mean_penalty"]
        i = lookup[(method, "incorrect")]["mean_penalty"]
        print(f"{method:<16} {c:+14.6f} {i:+16.6f} {c-i:+12.6f}")
    print(f"\nSaved CSV + PNG + PDF to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
