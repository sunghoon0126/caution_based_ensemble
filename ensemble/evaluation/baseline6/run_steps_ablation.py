"""Baseline 6 Test 1: one 8-step SGD trajectory, evaluate steps 1/2/4/8.

Uses the original 512-candidate RM pool to define the SAME fixed minimum dummy
reward for every N, while adapting only the first 256 candidates per problem.
Each candidate starts from the frozen original RM parameters. Completed rows
are flushed and validated for safe resume. No generator or base-RM re-scoring.
"""

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.reward_model_adapter import RewardModelAdapter


DEFAULT_STEPS = (1, 2, 4, 8)


class StepResumeWriter(ResumeWriter):
    """Support already-written step_scores.jsonl rows without reward_after.

    The original ResumeWriter requires reward_after when verifying completed
    rows. Earlier Test 1 output stored scores_by_step instead. Normalize
    those rows in memory while streaming; never rewrite or discard results.
    """

    def __init__(self, path, config):
        super().__init__(path, config)
        step_keys = tuple(str(step) for step in config["steps"])
        original_rows = self.existing

        def normalize():
            for row in original_rows:
                values = row.get("scores_by_step")
                if not isinstance(values, dict):
                    raise ValueError("Existing Test 1 row lacks scores_by_step")
                for key in step_keys:
                    if key not in values:
                        raise ValueError(f"Existing Test 1 row missing step {key}")
                    finite(values[key])
                if "reward_after" in row:
                    if not math.isclose(
                        finite(row["reward_after"]),
                        finite(values[step_keys[-1]]),
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        raise ValueError("Existing row reward_after != last checkpoint")
                else:
                    row["reward_after"] = finite(values[step_keys[-1]])
                yield row

        self.existing = normalize()


def adapted_checkpoints(model, inputs, target, lr, steps):
    """Stateless, full-model SGD; record scores along a single trajectory."""
    import torch
    from torch.func import functional_call

    model.eval()
    base = dict(model.named_parameters())
    params = {k: v.detach().requires_grad_(True) for k, v in base.items()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    target = finite(target)

    def forward():
        result = functional_call(model, (params, buffers), (), inputs, strict=True)
        logits = result.logits if hasattr(result, "logits") else result
        if logits.numel() != 1:
            raise ValueError("Reward model must return one scalar")
        return logits.reshape(()).float()

    scores = {}
    with torch.enable_grad():
        reward = forward()
        before = finite(reward.detach().item())
        for step in range(1, max(steps) + 1):
            loss = 0.5 * (reward - target).square()
            finite(loss.detach().item())
            grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
            updated = {}
            with torch.no_grad():
                for (name, tensor), grad in zip(params.items(), grads):
                    if grad is None:
                        updated[name] = tensor.detach().requires_grad_(True)
                    else:
                        value = tensor - lr * grad
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(f"Nonfinite parameter {name} at step {step}")
                        updated[name] = value.detach().requires_grad_(True)
            params = updated
            del loss, grads
            reward = forward()
            if step in steps:
                scores[str(step)] = finite(reward.detach().item())
    return before, scores


def make_adapter(args, first_row):
    adapter = RewardModelAdapter(
        args.reward_model, args.device, args.dtype, args.reward_backend,
        args.revision, args.max_length,
    )
    if first_row.get("reward_model_path") != args.reward_model:
        raise ValueError("The original scores use a different reward model")
    if first_row.get("reward_scoring_config") != adapter.settings:
        raise ValueError(
            f"Scoring settings mismatch: original={first_row.get('reward_scoring_config')}, "
            f"current={adapter.settings}"
        )
    recorded_revision = first_row.get("reward_model_revision")
    resolved_revision = getattr(adapter.model.config, "_commit_hash", None)
    if recorded_revision and recorded_revision != resolved_revision:
        raise ValueError("Reward model checkpoint revision differs")
    return adapter


def score(args):
    steps = tuple(args.steps)
    cfg = {
        "ablation": "baseline6_sgd_steps_v1",
        "learning_rate": args.learning_rate,
        "steps": list(steps),
        "optimizer": "sgd",
        "momentum": 0,
        "weight_decay": 0,
        "loss": "half_mse",
        "scope": "full",
        "dummy_mode": "fixed_min",
        "dummy_calibration_pool": args.dummy_pool,
        "scored_candidates_per_problem": args.scored_candidates,
        "max_problems": args.max_problems,
        "original_path": str(args.original.resolve()),
        "reward_model": args.reward_model,
        "device": args.device,
        "dtype": args.dtype,
        "backend": args.reward_backend,
        "revision": args.revision,
        "max_length": args.max_length,
    }
    writer = StepResumeWriter(args.output / "step_scores.jsonl", cfg)
    adapter = None
    expected_settings = None
    seen = 0
    new = 0
    total = args.max_problems * args.scored_candidates
    started = time.perf_counter()
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, desc="Test 1 · SGD checkpoints", unit="cand", dynamic_ncols=True)
    except ImportError:
        bar = None
    try:
        for problem_i, full_group in enumerate(groups(args.original, args.dummy_pool)):
            if problem_i >= args.max_problems:
                break
            dummy = min(finite(r["reward_model_score"]) for r in full_group)
            context = {"dummy_reward": dummy, "pool_hash": digest(full_group)}
            for row in full_group[:args.scored_candidates]:
                if row.get("reward_model_path") != args.reward_model:
                    raise ValueError("Mixed reward models in original score file")
                current_settings = (
                    row.get("reward_scoring_config"),
                    row.get("reward_model_revision"),
                )
                if expected_settings is not None and current_settings != expected_settings:
                    raise ValueError("Mixed original RM settings")
                expected_settings = current_settings
                completed, provenance = writer.completed(row, context)
                seen += 1
                if completed:
                    if bar is not None:
                        bar.update(1)
                    continue
                if adapter is None:
                    print("Loading original reward model...", flush=True)
                    adapter = make_adapter(args, row)
                    print("Reward model ready; scoring checkpoints", flush=True)
                try:
                    before, scores = adapted_checkpoints(
                        adapter.model, adapter.inputs(row["prompt"], row["response"]),
                        dummy, args.learning_rate, steps,
                    )
                    original = finite(row["reward_model_score"])
                    if not math.isclose(before, original, rel_tol=1e-4, abs_tol=1e-4):
                        raise ValueError(
                            f"Original RM score mismatch: before={before}, stored={original}"
                        )
                    writer.write({
                        **row,
                        **provenance,
                        "dummy_reward": dummy,
                        "dummy_calibration_pool": args.dummy_pool,
                        "reward_before": before,
                        "reward_after": scores[str(steps[-1])],
                        "scores_by_step": scores,
                        "finite": True,
                    })
                except Exception as exc:
                    failure_path = args.output / "failure.json"
                    failure_path.write_text(json.dumps({
                        "candidate_id": row.get("candidate_id"),
                        "problem_index": problem_i,
                        "error": repr(exc),
                    }, indent=2))
                    raise
                new += 1
                if bar is not None:
                    bar.update(1)
            if (problem_i + 1) % 10 == 0 or problem_i == 0:
                print(
                    f"Problems {problem_i + 1}/{args.max_problems}; "
                    f"candidates {seen}/{total}; newly scored {new}; "
                    f"elapsed {(time.perf_counter() - started)/3600:.2f} h",
                    flush=True,
                )
        writer.finish()
        if seen != total:
            raise ValueError(f"Expected {total} candidate records, visited {seen}")
    finally:
        writer.close()
        if bar is not None:
            bar.close()
    print("Step scoring complete", flush=True)


def evaluate(args):
    ns = [2**i for i in range(args.scored_candidates.bit_length())]
    ns = [n for n in ns if n <= args.scored_candidates]
    methods = ["vanilla"] + [f"step_{k}" for k in args.steps]
    counts = {(method, n): 0 for method in methods for n in ns}
    flip = {(method, n): Counter() for method in methods if method != "vanilla" for n in ns}
    num_problems = 0
    for group in groups(args.output / "step_scores.jsonl", args.scored_candidates):
        num_problems += 1
        indices = {method: -1 for method in methods}
        best = {method: float("-inf") for method in methods}
        for i, row in enumerate(group, start=1):
            truth = row["correct"]
            if type(truth) not in (bool, int) or truth not in (0, 1):
                raise ValueError(f"Invalid correctness at candidate {row['candidate_id']}")
            if row.get("dummy_calibration_pool") != args.dummy_pool or not row.get("finite"):
                raise ValueError("Mismatched dummy calibration or failed score")
            scores = {"vanilla": finite(row["reward_model_score"])}
            for step in args.steps:
                scores[f"step_{step}"] = finite(row["scores_by_step"][str(step)])
            for method, reward in scores.items():
                if reward > best[method]:
                    best[method] = reward
                    indices[method] = i - 1
            if i not in ns:
                continue
            selected = {method: int(group[indices[method]]["correct"]) for method in methods}
            for method in methods:
                counts[(method, i)] += selected[method]
            for method in methods[1:]:
                if indices[method] == indices["vanilla"]:
                    flip[(method, i)]["same"] += 1
                elif not selected["vanilla"] and selected[method]:
                    flip[(method, i)]["I_to_C"] += 1
                elif selected["vanilla"] and not selected[method]:
                    flip[(method, i)]["C_to_I"] += 1
                else:
                    flip[(method, i)]["same_correctness"] += 1
    if num_problems != args.max_problems:
        raise ValueError(f"Expected {args.max_problems} problems; got {num_problems}")

    curves = []
    for method in methods:
        for n in ns:
            orig = counts[("vanilla", n)]
            current = counts[(method, n)]
            f = flip.get((method, n), Counter())
            curves.append({
                "method": method, "steps": 0 if method == "vanilla" else int(method[5:]),
                "N": n, "accuracy_percent": 100 * current / num_problems,
                "difference_vs_vanilla_pp": 100 * (current - orig) / num_problems,
                "flip_rate_percent": 100 * (num_problems - f["same"]) / num_problems if method != "vanilla" else 0,
                "I_to_C": f["I_to_C"], "C_to_I": f["C_to_I"],
                "num_problems": num_problems,
                "dummy_calibration_pool": args.dummy_pool,
            })
    summaries = []
    for method in methods:
        series = [row for row in curves if row["method"] == method]
        peak = max(series, key=lambda x: x["accuracy_percent"])
        final = series[-1]
        summaries.append({
            "method": method,
            "peak_n": peak["N"], "peak_accuracy_percent": peak["accuracy_percent"],
            "final_n": final["N"], "final_accuracy_percent": final["accuracy_percent"],
            "degradation_pp": peak["accuracy_percent"] - final["accuracy_percent"],
            "final_difference_vs_vanilla_pp": final["difference_vs_vanilla_pp"],
            "final_flip_rate_percent": final["flip_rate_percent"],
        })

    out = args.output / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("curves.csv", curves), ("summary.csv", summaries)):
        with (out / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        series = [row for row in curves if row["method"] == method]
        ax.plot(ns, [row["accuracy_percent"] for row in series], marker="o", label=method)
    ax.set(xlabel="Selection budget N", ylabel="Accuracy (%)",
           title=f"Baseline 6 SGD Steps · {num_problems} problems · LR={args.learning_rate:g}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_n.png", dpi=200)
    fig.savefig(out / "accuracy_vs_n.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods[1:]:
        series = [row for row in curves if row["method"] == method]
        ax.plot(ns, [row["flip_rate_percent"] for row in series], marker="o", label=method)
    ax.set(xlabel="Selection budget N", ylabel="Selection Flip Rate (%)",
           title="SGD Step Ablation · Flip vs Original RM")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "flip_rate_vs_n.png", dpi=200)
    plt.close(fig)

    print("\n=== Test 1 Summary ===", flush=True)
    for row in summaries:
        print(row, flush=True)
    print(f"Evaluation saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--dummy-pool", type=int, default=512)
    parser.add_argument("--scored-candidates", type=int, default=256)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--reward-model", default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--reward-backend", default="auto")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or args.max_problems <= 0 or args.dummy_pool <= 0
            or args.scored_candidates <= 0 or args.scored_candidates > args.dummy_pool
            or args.scored_candidates & (args.scored_candidates - 1)):
        parser.error("Invalid learning rate, problem count or sample sizes")
    if args.steps != sorted(set(args.steps)) or not args.steps or args.steps[0] <= 0:
        parser.error("Steps must be unique, positive and increasing")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "test1_config.json"
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(args).items() if k != "evaluate_only"}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != config:
            raise ValueError("Experiment configuration changed; use a different --output")
    else:
        manifest_path.write_text(json.dumps(config, indent=2))
    if not args.evaluate_only:
        score(args)
    evaluate(args)


if __name__ == "__main__":
    main()
