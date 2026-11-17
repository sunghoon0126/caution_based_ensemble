"""Baseline 6 T3-A: constant dummy, SGD 1-step, candidates share one backward.

For every candidate, freeze d_i = original_RM_score_i - c.
Calibrate a global c0 from the old minimum-dummy experiment's reward-gradient
norms (Top-16 within first 256). Score c=c0 and c=2*c0 with one backward pass
and two stateless forward passes. This is exploratory on the current test pool.
"""

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.reward_model_adapter import RewardModelAdapter


MODES = ("c0", "2c0")


def calibrate_c0(args):
    """Match a reference Top-K first-order mean penalty per problem.

    Old minimum-dummy loss gradient satisfies
        ||grad(L)|| = (original_score - minimum) * ||grad(reward)||.
    Thus G_i = ||grad(reward_i)||^2 is obtained without a new backward.
    c_x = sum_topK (r_i-min)*G_i / sum_topK G_i.
    One global c0 is the median of c_x across problems.
    """
    if args.base_constant is not None:
        c0 = finite(args.base_constant)
        if c0 <= 0:
            raise ValueError("--base-constant must be positive")
        return c0, {"method": "user", "c0": c0}
    if args.reference_bp is None:
        raise ValueError("Provide --reference-bp or --base-constant")
    values = []
    for index, (original, adapted) in enumerate(zip_longest(
        groups(args.original, args.dummy_pool),
        groups(args.reference_bp, args.dummy_pool),
    )):
        if index == args.max_problems:
            break
        if original is None or adapted is None:
            raise ValueError("Original and reference BP problem pools differ")
        minimum = min(finite(row["reward_model_score"]) for row in original)
        top = sorted(
            range(args.scored_candidates),
            key=lambda i: finite(original[i]["reward_model_score"]),
            reverse=True,
        )[:args.top_k]
        numerator = denominator = 0.0
        for i in top:
            source, bp = original[i], adapted[i]
            if source["candidate_id"] != bp["candidate_id"] or bp.get("finite") is not True:
                raise ValueError(f"Reference BP candidate mismatch at problem {index}")
            if not math.isclose(finite(bp["dummy_reward"]), minimum, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Reference BP dummy reward differs from 512-pool minimum")
            reward = finite(source["reward_model_score"])
            if not math.isclose(finite(bp["reward_before"]), reward, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("Reference BP original score mismatch")
            gap = reward - minimum
            if gap <= 1e-5:
                continue
            grad_loss_norm = finite(bp["gradient_norm"])
            G = (grad_loss_norm / gap) ** 2
            if not math.isfinite(G):
                raise ValueError("Nonfinite inferred reward gradient norm")
            numerator += gap * G
            denominator += G
        if denominator > 0:
            values.append(numerator / denominator)
    if len(values) != args.max_problems:
        raise ValueError(
            f"Only {len(values)} of {args.max_problems} calibration problems were valid"
        )
    c0 = finite(statistics.median(values))
    if c0 <= 0:
        raise ValueError("Calibrated c0 must be positive")
    return c0, {
        "method": "median_of_topk_gradient_weighted_reward_gaps",
        "c0": c0, "num_problems": len(values), "top_k": args.top_k,
        "mean_problem_c": statistics.mean(values),
        "min_problem_c": min(values), "max_problem_c": max(values),
        "source_reference_bp": str(args.reference_bp.resolve()),
        "note": "Exploratory c0 calibrated on the same GSM8K test problem pool",
    }


def score_candidate(model, inputs, constant_map, learning_rate):
    """Exact first SGD step for d_i=r0-c, sharing one grad(r0)."""
    import torch
    from torch.func import functional_call

    model.eval()
    parameters = {
        name: tensor.detach().requires_grad_(True)
        for name, tensor in model.named_parameters()
    }
    buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}

    def scalar(output):
        logits = output.logits if hasattr(output, "logits") else output
        if logits.numel() != 1:
            raise ValueError("Expected scalar reward")
        return logits.reshape(()).float()

    with torch.enable_grad():
        reward = scalar(functional_call(
            model, (parameters, buffers), (), inputs, strict=True
        ))
        before = finite(reward.detach().item())
        reward_grads = torch.autograd.grad(
            reward, list(parameters.values()), allow_unused=True
        )

    scores = {}
    with torch.no_grad():
        for mode, c in constant_map.items():
            adapted = {}
            for (name, tensor), grad in zip(parameters.items(), reward_grads):
                if grad is None:
                    adapted[name] = tensor.detach()
                else:
                    adapted[name] = tensor.detach() - learning_rate * c * grad
                    if not torch.isfinite(adapted[name]).all():
                        raise FloatingPointError(f"Nonfinite updated weight: {name}")
            new_reward = scalar(functional_call(
                model, (adapted, buffers), (), inputs, strict=True
            ))
            scores[mode] = finite(new_reward.item())
    return before, scores


def load_adapter(args, row):
    adapter = RewardModelAdapter(
        args.reward_model, args.device, args.dtype, args.reward_backend,
        args.revision, args.max_length,
    )
    if row.get("reward_model_path") != args.reward_model:
        raise ValueError("Original score RM model mismatch")
    if row.get("reward_scoring_config") != adapter.settings:
        raise ValueError("Original and new RM tokenization/scoring settings differ")
    recorded = row.get("reward_model_revision")
    actual = getattr(adapter.model.config, "_commit_hash", None)
    if recorded and actual != recorded:
        raise ValueError("Resolved RM revision mismatch")
    return adapter


def run_scoring(args, constants, run_config):
    writer = ResumeWriter(args.output / "constant_scores.jsonl", run_config)
    adapter = None
    total = args.max_problems * args.scored_candidates
    seen = newly_scored = 0
    started = time.perf_counter()
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, unit="candidate", desc="T3 SGD 1-step constants",
                   dynamic_ncols=True)
    except ImportError:
        bar = None
    try:
        for problem_idx, full_group in enumerate(groups(args.original, args.dummy_pool)):
            if problem_idx >= args.max_problems:
                break
            minimum = min(finite(row["reward_model_score"]) for row in full_group)
            context = {"dummy_reward": minimum, "pool_hash": digest(full_group)}
            for row in full_group[:args.scored_candidates]:
                completed, provenance = writer.completed(row, context)
                seen += 1
                if completed:
                    if bar is not None:
                        bar.update(1)
                    continue
                if adapter is None:
                    print("Loading original Reward Model...", flush=True)
                    adapter = load_adapter(args, row)
                    print("Reward Model ready", flush=True)
                try:
                    before, updated = score_candidate(
                        adapter.model,
                        adapter.inputs(row["prompt"], row["response"]),
                        constants, args.learning_rate,
                    )
                    if not math.isclose(
                        before, finite(row["reward_model_score"]),
                        rel_tol=1e-4, abs_tol=1e-4,
                    ):
                        raise ValueError("Original RM score mismatch")
                    writer.write({
                        **row, **provenance,
                        "dummy_mode": "original_minus_constant",
                        "dummy_calibration_pool": args.dummy_pool,
                        "full_pool_minimum": minimum,
                        "reward_before": before,
                        "reward_after": updated["2c0"],  # Required by ResumeWriter.
                        "scores_by_constant": updated,
                        "finite": True,
                    })
                except Exception as error:
                    (args.output / "failure.json").write_text(json.dumps({
                        "problem_index": problem_idx,
                        "candidate_id": row.get("candidate_id"),
                        "exception": repr(error),
                    }, indent=2))
                    raise
                newly_scored += 1
                if bar is not None:
                    bar.update(1)
            if problem_idx == 0 or (problem_idx + 1) % 10 == 0:
                print(
                    f"Problems {problem_idx+1}/{args.max_problems}; "
                    f"visited {seen:,}/{total:,}; new {newly_scored:,}; "
                    f"elapsed {(time.perf_counter()-started)/3600:.2f} h",
                    flush=True,
                )
        writer.finish()
        if seen != total:
            raise ValueError(f"Expected {total} candidate rows, saw {seen}")
    finally:
        writer.close()
        if bar is not None:
            bar.close()


def evaluate(args, constants):
    methods = ("vanilla", "sgd_min_step1", *MODES)
    n_values = [2 ** i for i in range(args.scored_candidates.bit_length())
                if 2 ** i <= args.scored_candidates]
    counts = {(method, n): Counter() for method in methods for n in n_values}
    penalty_values = {mode: {0: [], 1: []} for mode in MODES}
    num_problems = 0
    for own, reference in zip_longest(
        groups(args.output / "constant_scores.jsonl", args.scored_candidates),
        groups(args.sgd_reference, args.scored_candidates),
    ):
        if own is None or reference is None:
            raise ValueError("T3 and Test 1 problem counts differ")
        num_problems += 1
        best = {name: float("-inf") for name in methods}
        selected_index = {name: -1 for name in methods}
        for i, (row, sgd) in enumerate(zip(own, reference), 1):
            if row["candidate_id"] != sgd["candidate_id"]:
                raise ValueError(f"T3 and Test 1 candidate mismatch at problem {num_problems}")
            if row.get("finite") is not True:
                raise ValueError("Nonfinite/failed T3 candidate")
            if row.get("dummy_calibration_pool") != args.dummy_pool:
                raise ValueError("Dummy calibration pool mismatch")
            truth = row["correct"]
            if type(truth) not in (bool, int) or truth not in (0, 1):
                raise ValueError("Invalid candidate correctness")
            original_score = finite(row["reward_model_score"])
            scores = {
                "vanilla": original_score,
                "sgd_min_step1": finite(sgd["scores_by_step"]["1"]),
                "c0": finite(row["scores_by_constant"]["c0"]),
                "2c0": finite(row["scores_by_constant"]["2c0"]),
            }
            for mode in MODES:
                penalty_values[mode][int(truth)].append(original_score - scores[mode])
            for method, value in scores.items():
                if value > best[method]:
                    best[method], selected_index[method] = value, i-1
            if i not in n_values:
                continue
            vanilla_idx = selected_index["vanilla"]
            vanilla_correct = int(own[vanilla_idx]["correct"])
            for method in methods:
                new_idx = selected_index[method]
                current = counts[(method, i)]
                correct = int(own[new_idx]["correct"])
                current["correct"] += correct
                if new_idx != vanilla_idx:
                    current["flips"] += 1
                    if vanilla_correct == 0 and correct == 1:
                        current["I_to_C"] += 1
                    elif vanilla_correct == 1 and correct == 0:
                        current["C_to_I"] += 1
    if num_problems != args.max_problems:
        raise ValueError(f"Expected {args.max_problems} problems, got {num_problems}")

    curves, summary = [], []
    for method in methods:
        points = []
        for n in n_values:
            stat = counts[(method, n)]
            base = counts[("vanilla", n)]
            row = {
                "method": method, "N": n,
                "accuracy_percent": 100 * stat["correct"] / num_problems,
                "difference_vs_vanilla_pp": 100 * (stat["correct"] - base["correct"]) / num_problems,
                "flip_rate_percent": 100 * stat["flips"] / num_problems,
                "I_to_C": stat["I_to_C"], "C_to_I": stat["C_to_I"],
                "num_problems": num_problems,
            }
            points.append(row)
            curves.append(row)
        peak = max(points, key=lambda row: row["accuracy_percent"])
        last = points[-1]
        summary.append({
            "method": method, "peak_n": peak["N"],
            "peak_accuracy_percent": peak["accuracy_percent"],
            "final_n": last["N"], "final_accuracy_percent": last["accuracy_percent"],
            "degradation_pp": peak["accuracy_percent"] - last["accuracy_percent"],
            "final_delta_pp": last["difference_vs_vanilla_pp"],
            "final_flip_rate_percent": last["flip_rate_percent"],
            "final_I_to_C": last["I_to_C"], "final_C_to_I": last["C_to_I"],
        })
    output = args.output / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("curves.csv", curves), ("summary.csv", summary)):
        with (output / name).open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    penalty_summary = []
    import numpy as np
    for mode in MODES:
        for label, subset in (("correct", penalty_values[mode][1]),
                              ("incorrect", penalty_values[mode][0])):
            values = np.asarray(subset, dtype=float)
            penalty_summary.append({
                "method": mode, "group": label, "count": len(values),
                "mean_penalty": float(np.mean(values)),
                "median_penalty": float(np.median(values)),
            })
    with (output / "penalty_summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(penalty_summary[0]))
        writer.writeheader()
        writer.writerows(penalty_summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        curve = [row for row in curves if row["method"] == method]
        ax.plot(n_values, [row["accuracy_percent"] for row in curve],
                marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values, [str(n) for n in n_values])
    ax.set(xlabel="Best-of-N", ylabel="Accuracy (%)",
           title=f"T3: Original score minus constant (c0={constants['c0']:.5g})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_n.png", dpi=200)
    fig.savefig(output / "accuracy_vs_n.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods[1:]:
        curve = [row for row in curves if row["method"] == method]
        ax.plot(n_values, [row["flip_rate_percent"] for row in curve],
                marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values, [str(n) for n in n_values])
    ax.set(xlabel="Best-of-N", ylabel="Selection flip vs Vanilla RM (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "flip_rate_vs_n.png", dpi=200)
    plt.close(fig)
    print("\n=== Test 3 Constant SGD Results ===", flush=True)
    for row in summary:
        print(row, flush=True)
    print(f"Saved: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reference-bp", type=Path,
                        help="Original full-512 minimum-dummy BP scores used to calibrate c0")
    parser.add_argument("--sgd-reference", type=Path, required=True,
                        help="Test 1 step_scores.jsonl on the same 500x256 candidates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--dummy-pool", type=int, default=512)
    parser.add_argument("--scored-candidates", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--base-constant", type=float,
                        help="Optional manual c0. Otherwise use --reference-bp.")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--reward-model",
                        default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--reward-backend", default="auto")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if (args.max_problems <= 0 or args.dummy_pool <= 0
        or args.scored_candidates <= 0 or args.scored_candidates > args.dummy_pool
        or args.scored_candidates & (args.scored_candidates - 1)
        or args.top_k <= 0 or args.top_k > args.scored_candidates
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Invalid problem, candidate, top-K or SGD settings")

    args.output.mkdir(parents=True, exist_ok=True)
    c0, calibration = calibrate_c0(args)
    constant_map = {"c0": c0, "2c0": 2 * c0}
    print(f"c0={c0:.9g}; 2c0={2*c0:.9g}", flush=True)
    manifest = {
        "study": "baseline6_original_minus_constant_sgd_1step",
        "inputs": {
            k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items() if k != "evaluate_only"
        },
        "constant_map": constant_map,
        "calibration": calibration,
    }
    manifest_file = args.output / "t3_config.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("Changed T3 configuration; use a new output directory")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2))
    config = {
        "mode": "original_minus_constant",
        "optimizer": "sgd", "num_steps": 1,
        "learning_rate": args.learning_rate,
        "constant_map": constant_map,
        "reward_model": args.reward_model,
        "dtype": args.dtype, "device": args.device,
        "backend": args.reward_backend,
        "revision": args.revision, "max_length": args.max_length,
        "dummy_pool": args.dummy_pool,
        "scored_candidates": args.scored_candidates,
        "max_problems": args.max_problems,
        "source_original": str(args.original.resolve()),
    }
    if not args.evaluate_only:
        run_scoring(args, constant_map, config)
    evaluate(args, constant_map)


if __name__ == "__main__":
    main()
