"""Read-only analysis of Baseline 6 multi-step SGD results.

Three analyses:
1. Mean/median penalty vs step,
2. Correct vs incorrect penalty (pooled and within-problem),
3. Best-of-N selection flips: I->C, C->I, C->C, I->I.

Consumes Test 1's step_scores.jsonl. No GPU or model loading.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ensemble.evaluation.baseline6.common import finite, groups


def write_csv(path, records):
    if not records:
        raise ValueError(f"No data for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def describe(a, step, group):
    data = np.asarray(a, dtype=np.float64)
    if not len(data):
        raise ValueError(f"Empty penalty group: step={step}, group={group}")
    return {
        "step": step,
        "group": group,
        "count": len(data),
        "mean_penalty": float(np.mean(data)),
        "median_penalty": float(np.median(data)),
        "std_penalty": float(np.std(data)),
        "p10_penalty": float(np.percentile(data, 10)),
        "p90_penalty": float(np.percentile(data, 90)),
        "fraction_positive": float(np.mean(data > 1e-9)),
        "fraction_negative": float(np.mean(data < -1e-9)),
        "fraction_near_zero": float(np.mean(np.abs(data) <= 1e-9)),
    }


def plot_line(rows, output, name, ylabel, title, ykeys, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["step"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for field, label in zip(ykeys, labels):
        y = [float(row[field]) for row in rows]
        ax.plot(x, y, "o-", label=label)
    ax.set_xticks(x)
    ax.set_xlabel("SGD steps (LR=1e-6)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=200)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plot_penalty_global(penalty_rows, output, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [next(r for r in penalty_rows if r["step"] == k and r["group"] == "all") for k in steps]
    for stat, title in [
        ("mean_penalty", "Mean Reward Penalty vs SGD Steps"),
        ("median_penalty", "Median Reward Penalty vs SGD Steps"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for group, label in [("all", "All candidates"),
                             ("correct", "Correct"),
                             ("incorrect", "Incorrect")]:
            subset = [next(r for r in penalty_rows if r["step"] == k and r["group"] == group) for k in steps]
            ax.plot(steps, [r[stat] for r in subset], marker="o", label=label)
        ax.set_xticks(steps)
        ax.set_xlabel("SGD Steps (LR=1e-6)")
        ax.set_ylabel("Reward penalty = original score - updated score")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fname = "mean_penalty_vs_steps" if stat == "mean_penalty" else "median_penalty_vs_steps"
        fig.savefig(output / f"{fname}.png", dpi=200)
        plt.close(fig)

    plot_line(
        all_rows, output, "mean_vs_median_penalty",
        "Reward penalty", "Mean and Median Reward Penalty vs SGD Steps",
        ["mean_penalty", "median_penalty"], ["Mean", "Median"],
    )


def plot_paired(paired_summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["step"] for row in paired_summary]
    mean = [row["mean_correct_minus_incorrect"] for row in paired_summary]
    median = [row["median_correct_minus_incorrect"] for row in paired_summary]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, mean, "o-", label="Mean paired difference")
    ax.plot(x, median, "s--", label="Median paired difference")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks(x)
    ax.set_xlabel("SGD Steps")
    ax.set_ylabel("Correct penalty - Incorrect penalty")
    ax.set_title("Within-problem Correct vs Incorrect Penalty")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "within_problem_penalty_diff_vs_steps.png", dpi=200)
    plt.close(fig)


def plot_flips(flip_rows, output, steps, final_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [next(r for r in flip_rows if r["step"] == k and r["N"] == final_n) for k in steps]
    x = np.arange(len(steps), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    ax.bar(x - width / 2, [r["I_to_C"] for r in rows], width, label="Incorrect → Correct", color="tab:green")
    ax.bar(x + width / 2, [r["C_to_I"] for r in rows], width, label="Correct → Incorrect", color="tab:red")
    ax.set_xticks(x, [str(k) for k in steps])
    ax.set_xlabel("SGD Steps")
    ax.set_ylabel("Number of problems")
    ax.set_title(f"Correctness Flips vs SGD Steps (N={final_n})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "correctness_flips_vs_steps.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(steps, [r["flip_rate_percent"] for r in rows], "o-", label="Selection Flip Rate")
    ax.set_xticks(steps)
    ax.set_xlabel("SGD Steps")
    ax.set_ylabel("Flip rate (%)")
    ax.set_title(f"Top-1 Selection Flip vs SGD Steps (N={final_n})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "selection_flip_rate_vs_steps.png", dpi=200)
    plt.close(fig)


def verify_against_existing_curves(rows, path):
    if path is None:
        return
    reference = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference[(row["method"], int(row["N"]))] = (
                float(row["accuracy_percent"]),
                float(row["flip_rate_percent"]),
            )
    for row in rows:
        key = (f"step_{row['step']}", row["N"])
        if key not in reference:
            raise ValueError(f"Missing reference curve: {key}")
        expected_accuracy, expected_flip = reference[key]
        if (abs(expected_accuracy - row["accuracy_percent"]) > 1e-7 or
                abs(expected_flip - row["flip_rate_percent"]) > 1e-7):
            raise ValueError(
                f"Mismatch with Test 1 evaluation for {key}: "
                f"expected {(expected_accuracy, expected_flip)}, "
                f"got {(row['accuracy_percent'], row['flip_rate_percent'])}"
            )
    print("Verified all step/N accuracy and flip rates against existing curves.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Test 1 step_scores.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--scored-candidates", type=int, default=256)
    parser.add_argument("--expected-problems", type=int, default=500)
    parser.add_argument("--verify-curves", type=Path, help="Existing Test 1 evaluation/curves.csv")
    args = parser.parse_args()

    steps = args.steps
    if steps != sorted(set(steps)) or not steps or steps[0] < 1:
        parser.error("Steps must be positive unique increasing integers")
    nmax = args.scored_candidates
    if nmax <= 0 or nmax & (nmax - 1):
        parser.error("--scored-candidates must be a power of two")
    ns = [2 ** j for j in range(nmax.bit_length()) if 2 ** j <= nmax]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Candidate-weighted distribution: every candidate receives equal weight.
    penalty_values = {
        (step, group): []
        for step in steps for group in ("all", "correct", "incorrect")
    }
    # Problem-weighted paired comparisons: every eligible problem receives equal weight.
    paired_values = {step: [] for step in steps}
    per_problem_rows = []

    outcomes = {(step, n): Counter() for step in steps for n in ns}
    detail_rows = []
    problems = 0
    candidate_count = 0

    for group in groups(args.input, nmax):
        problems += 1
        candidate_count += len(group)
        correct = np.asarray([int(row["correct"]) for row in group], dtype=np.int8)
        if any(type(row["correct"]) not in (bool, int) or row["correct"] not in (0, 1) for row in group):
            raise ValueError(f"Invalid correctness in problem {problems}")
        original = np.asarray([finite(row["reward_model_score"]) for row in group], dtype=np.float64)
        before = np.asarray([finite(row["reward_before"]) for row in group], dtype=np.float64)
        if not np.allclose(original, before, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Before/original mismatch: problem {problems}")
        if any(row.get("finite") is not True for row in group):
            raise ValueError(f"Failed candidate in problem {problems}")
        dummy = finite(group[0]["dummy_reward"])
        if any(not math.isclose(finite(row["dummy_reward"]), dummy, rel_tol=0, abs_tol=1e-8) for row in group):
            raise ValueError(f"Inconsistent dummy in problem {problems}")

        original_winners = {n: int(np.argmax(original[:n])) for n in ns}

        for step in steps:
            updated = np.asarray([finite(row["scores_by_step"][str(step)]) for row in group], dtype=np.float64)
            penalty = original - updated
            penalty_values[(step, "all")].extend(penalty.tolist())
            penalty_values[(step, "correct")].extend(penalty[correct == 1].tolist())
            penalty_values[(step, "incorrect")].extend(penalty[correct == 0].tolist())

            cpen = penalty[correct == 1]
            ipen = penalty[correct == 0]
            if cpen.size and ipen.size:
                difference = float(cpen.mean() - ipen.mean())
                paired_values[step].append(difference)
                per_problem_rows.append({
                    "dataset_index": group[0]["dataset_index"],
                    "instance_id": group[0]["instance_id"],
                    "step": step,
                    "num_correct": len(cpen), "num_incorrect": len(ipen),
                    "mean_correct_penalty": float(cpen.mean()),
                    "mean_incorrect_penalty": float(ipen.mean()),
                    "difference_correct_minus_incorrect": difference,
                })

            for n in ns:
                winner = int(np.argmax(updated[:n]))
                base = original_winners[n]
                base_truth = int(correct[base])
                after_truth = int(correct[winner])
                counter = outcomes[(step, n)]
                counter["original_correct"] += base_truth
                counter["updated_correct"] += after_truth
                if winner == base:
                    category = "same"
                elif base_truth == 0 and after_truth == 1:
                    category = "I_to_C"
                elif base_truth == 1 and after_truth == 0:
                    category = "C_to_I"
                elif base_truth == 1 and after_truth == 1:
                    category = "C_to_C"
                else:
                    category = "I_to_I"
                counter[category] += 1
                if winner != base and n == nmax:
                    detail_rows.append({
                        "dataset_index": group[0]["dataset_index"],
                        "instance_id": group[0]["instance_id"],
                        "step": step, "N": n, "category": category,
                        "original_candidate_id": group[base]["candidate_id"],
                        "updated_candidate_id": group[winner]["candidate_id"],
                        "original_correct": base_truth,
                        "updated_correct": after_truth,
                        "original_selected_score": float(original[base]),
                        "original_selected_penalty": float(penalty[base]),
                        "updated_selected_original_score": float(original[winner]),
                        "updated_selected_penalty": float(penalty[winner]),
                    })

        if problems % 50 == 0:
            print(f"Analyzed {problems} problems ({candidate_count:,} candidates)", flush=True)

    if problems != args.expected_problems:
        raise ValueError(f"Expected {args.expected_problems} problems, found {problems}")
    penalty_summary = []
    for step in steps:
        for group in ("all", "correct", "incorrect"):
            penalty_summary.append(describe(penalty_values[(step, group)], step, group))

    paired_summary = []
    for step in steps:
        differences = np.asarray(paired_values[step], dtype=np.float64)
        if not len(differences):
            raise ValueError(f"No problems with both correct and incorrect candidates for step {step}")
        paired_summary.append({
            "step": step,
            "eligible_problems": len(differences),
            "mean_correct_minus_incorrect": float(differences.mean()),
            "median_correct_minus_incorrect": float(np.median(differences)),
            "fraction_correct_penalty_larger": float(np.mean(differences > 0)),
            "fraction_incorrect_penalty_larger": float(np.mean(differences < 0)),
        })

    flip_summary = []
    for step in steps:
        for n in ns:
            c = outcomes[(step, n)]
            total = sum(c[category] for category in ("same", "I_to_C", "C_to_I", "C_to_C", "I_to_I"))
            if total != problems:
                raise ValueError(f"Flip counts do not sum to total problems: step {step}, N {n}")
            flip_count = total - c["same"]
            net = c["I_to_C"] - c["C_to_I"]
            if net != c["updated_correct"] - c["original_correct"]:
                raise ValueError(f"Net correctness change mismatch: step {step}, N {n}")
            flip_summary.append({
                "step": step, "N": n, "num_problems": problems,
                "original_accuracy_percent": 100 * c["original_correct"] / problems,
                "updated_accuracy_percent": 100 * c["updated_correct"] / problems,
                "difference_vs_vanilla_pp": 100 * net / problems,
                "flip_count": flip_count,
                "flip_rate_percent": 100 * flip_count / problems,
                "I_to_C": c["I_to_C"], "C_to_I": c["C_to_I"],
                "C_to_C": c["C_to_C"], "I_to_I": c["I_to_I"],
                "unchanged": c["same"],
                "net_correct": net,
            })

    verify_against_existing_curves(flip_summary, args.verify_curves)
    out = args.output_dir
    write_csv(out / "penalty_summary_by_step.csv", penalty_summary)
    write_csv(out / "per_problem_penalty_by_step.csv", per_problem_rows)
    write_csv(out / "paired_penalty_summary_by_step.csv", paired_summary)
    write_csv(out / "flip_by_step_and_n.csv", flip_summary)
    if detail_rows:
        write_csv(out / "selection_flip_details_n256.csv", detail_rows)

    plot_penalty_global(penalty_summary, out, steps)
    plot_paired(paired_summary, out)
    plot_flips(flip_summary, out, steps, nmax)

    print("\n=== Penalty Summary: All Candidates ===")
    for row in penalty_summary:
        if row["group"] == "all":
            print(f"Step {row['step']:>2}: mean={row['mean_penalty']:+.6f}, "
                  f"median={row['median_penalty']:+.6f}, "
                  f"reward decreased={100*row['fraction_positive']:.2f}%")
    print("\n=== Within-Problem Penalty: Correct - Incorrect ===")
    for row in paired_summary:
        print(f"Step {row['step']:>2}: mean diff={row['mean_correct_minus_incorrect']:+.6f}, "
              f"fraction(correct larger)={100*row['fraction_correct_penalty_larger']:.1f}%")
    print(f"\n=== Correctness Flips at N={nmax} ===")
    print("Steps | Flip(%) | I->C | C->I | C->C | I->I | Net")
    for step in steps:
        row = next(r for r in flip_summary if r["step"] == step and r["N"] == nmax)
        print(f"{step:>5} | {row['flip_rate_percent']:>7.2f} | "
              f"{row['I_to_C']:>4} | {row['C_to_I']:>4} | "
              f"{row['C_to_C']:>4} | {row['I_to_I']:>4} | {row['net_correct']:>+3}")
    print(f"\nSaved analysis CSVs and charts in {out}", flush=True)


if __name__ == "__main__":
    main()
