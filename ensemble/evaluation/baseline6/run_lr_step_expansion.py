"""Expand Baseline 6 SGD learning-rate/step sweep without lambda.

Evaluation ALWAYS selects the candidate using the actually adapted reward:
    argmax_i f_{theta_i^(K)}(x, y_i), i=1...N.
Each candidate and each LR trajectory starts from original theta0. The fixed
minimum dummy target is computed from all 512 original RM scores per prompt.

Reuse:
  Test 1: LR=1e-6, steps=1/2/4/8 (no new GPU scoring).
  Old LHS: its 10 exact LR/step pairs (no new GPU scoring).
Add:
  Full step checkpoints for each of several NEW LRs. Each LR requires only
  ONE trajectory up to max(steps), not an independent run for each checkpoint.
Two GPU workers split problems by index parity and use prefix-safe resume.

Output curves, summaries, tune/holdout comparison and LR-step heatmaps.
No lambda, penalty combination, or reward mixing is performed anywhere.
"""

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.reward_model_adapter import RewardModelAdapter
from ensemble.evaluation.baseline6.run_steps_ablation import adapted_checkpoints
from ensemble.evaluation.baseline6.run_lhs_sgd_sweep import (
    read_design as read_old_design,
    load_records as read_old_records,
)


DEFAULT_LRS = (5e-8, 1.5e-7, 3.5e-7, 7e-7)
DEFAULT_STEPS = (1, 2, 4, 8, 16)
TEST1_STEPS = (1, 2, 4, 8)


def load_config(out):
    d = json.loads((out / "lr_step_config.json").read_text(encoding="utf-8"))
    if d.get("version") != 1:
        raise ValueError("Invalid LR/steps experiment version")
    return d


def init(args):
    if args.original is None or args.sgd_reference is None or args.old_lhs is None:
        raise ValueError("init requires --original, --sgd-reference and --old-lhs")
    if not (args.pilot_problems >= 2 and
            args.dummy_pool >= args.scored_candidates > 0 and
            args.scored_candidates & (args.scored_candidates - 1) == 0 and
            args.lrs and args.lrs == sorted(set(args.lrs)) and
            all(math.isfinite(lr) and 0 < lr < 1e-6 for lr in args.lrs) and
            args.steps and args.steps == sorted(set(args.steps)) and args.steps[0] >= 1):
        raise ValueError("Invalid LR, step or data range")

    for p in (args.original, args.sgd_reference):
        if not p.is_file():
            raise FileNotFoundError(p)
    old = read_old_design(args.old_lhs)
    prev = old["settings"]
    expected = {
        "original": str(args.original.resolve()),
        "sgd_reference": str(args.sgd_reference.resolve()),
        "pilot_problems": args.pilot_problems,
        "dummy_pool": args.dummy_pool,
        "scored_candidates": args.scored_candidates,
        "reward_model": args.reward_model,
        "reward_backend": args.reward_backend,
        "dtype": args.dtype,
        "revision": args.revision,
        "max_length": args.max_length,
    }
    for key, value in expected.items():
        if prev.get(key) != value:
            raise ValueError(
                f"Old LHS settings mismatch ({key}): {prev.get(key)!r} vs {value!r}"
            )
    for rank in (0, 1):
        if not (args.old_lhs / f"rank_{rank}.jsonl").is_file():
            raise FileNotFoundError(
                f"Old LHS shard missing: {args.old_lhs / f'rank_{rank}.jsonl'}"
            )
    new_lrs = list(args.lrs)
    for lr in new_lrs:
        if lr == 1e-6:
            raise ValueError("LR=1e-6 is already included in Test 1")
    # A new LR that exactly coincides with an old LHS arm can still add more
    # step checkpoints. This is intentional; provenance stays separate.
    config = {
        "version": 1,
        "settings": expected,
        "old_lhs": str(args.old_lhs.resolve()),
        "lrs": new_lrs,
        "steps": list(args.steps),
        "split_seed": args.split_seed,
        "world_size": 2,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "lr_step_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Configuration differs; use another output directory")
    else:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    arms = []
    for source, lr, steps in (
        [("test1", 1e-6, TEST1_STEPS)]
        + [(f"old_lhs_{arm['id']}", arm["learning_rate"], (arm["steps"],))
           for arm in old["designs"]]
        + [(f"new_lr{idx:02d}", lr, args.steps)
           for idx, lr in enumerate(new_lrs)]
    ):
        for step in steps:
            arms.append({
                "source": source, "learning_rate": lr, "steps": step,
                "needs_gpu": source.startswith("new_lr"),
            })
    write_csv(args.output_dir / "all_arms.csv", arms)
    print(f"Initialized: {path}")
    print(f"Total configurations: {len(arms)} "
          f"({len(arms)-len(new_lrs)*len(args.steps)} reused, "
          f"{len(new_lrs)*len(args.steps)} new)")
    print(f"Total new SGD steps per candidate: {len(new_lrs)*max(args.steps)}")
    for idx, lr in enumerate(new_lrs):
        print(f"New trajectory LR {idx:02d} = {lr:.8g}, checkpoints {args.steps}")


def write_csv(path, records):
    if not records:
        raise ValueError(f"Cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def make_adapter(settings, device, first_row):
    adapter = RewardModelAdapter(
        settings["reward_model"], device, settings["dtype"],
        settings["reward_backend"], settings["revision"],
        settings["max_length"],
    )
    if first_row.get("reward_model_path") != settings["reward_model"]:
        raise ValueError("Reward model mismatch")
    if first_row.get("reward_scoring_config") != adapter.settings:
        raise ValueError("RM scoring configuration mismatch")
    checkpoint_revision = getattr(adapter.model.config, "_commit_hash", None)
    if first_row.get("reward_model_revision") and (
        first_row["reward_model_revision"] != checkpoint_revision
    ):
        raise ValueError("Reward model revision mismatch")
    return adapter


def worker(args):
    config = load_config(args.output_dir)
    s = config["settings"]
    if args.rank not in (0, 1) or config["world_size"] != 2:
        raise ValueError("Worker rank must be 0 or 1")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    output = args.output_dir / f"rank_{args.rank}.jsonl"
    writer = ResumeWriter(output, {
        "experiment": config, "rank": args.rank,
    })
    adapter = None
    own = sum(p % 2 == args.rank for p in range(s["pilot_problems"]))
    total = own * s["scored_candidates"]
    started = time.monotonic()
    processed = newly = 0
    print(f"[rank {args.rank}] Starting; GPU={args.device}; "
          f"problems={own}; candidates={total:,}; "
          f"learning-rates={config['lrs']}; steps={config['steps']}", flush=True)
    try:
        for p, full in enumerate(groups(Path(s["original"]), s["dummy_pool"])):
            if p >= s["pilot_problems"]:
                break
            if p % 2 != args.rank:
                continue
            dummy = min(finite(row["reward_model_score"]) for row in full)
            context = {"dummy_reward": dummy, "pool_hash": digest(full)}
            for row in full[:s["scored_candidates"]]:
                completed, provenance = writer.completed(row, context)
                processed += 1
                if not completed:
                    if adapter is None:
                        print(f"[rank {args.rank}] Loading reward model...", flush=True)
                        adapter = make_adapter(s, args.device, row)
                        print(f"[rank {args.rank}] Reward model ready", flush=True)
                    try:
                        inputs = adapter.inputs(row["prompt"], row["response"])
                        scores = {}
                        previous_before = None
                        for lr_idx, lr in enumerate(config["lrs"]):
                            before, checkpoints = adapted_checkpoints(
                                adapter.model, inputs, dummy, lr,
                                tuple(config["steps"]),
                            )
                            if previous_before is not None and not math.isclose(
                                previous_before, before, abs_tol=1e-5, rel_tol=1e-5
                            ):
                                raise ValueError("Original reward changed between LRs")
                            previous_before = before
                            scores[f"lr{lr_idx:02d}"] = checkpoints
                        original = finite(row["reward_model_score"])
                        if not math.isclose(
                            previous_before, original, abs_tol=1e-4, rel_tol=1e-4
                        ):
                            raise ValueError(
                                f"Original RM differs: stored={original}, "
                                f"recomputed={previous_before}"
                            )
                        last_key = f"lr{len(config['lrs'])-1:02d}"
                        writer.write({
                            **row, **provenance,
                            "dummy_reward": dummy,
                            "reward_before": previous_before,
                            "reward_after": scores[last_key][str(config["steps"][-1])],
                            "scores_by_lr": scores,
                            "finite": True,
                        })
                    except Exception as exc:
                        (args.output_dir / f"failure_rank_{args.rank}.json").write_text(
                            json.dumps({
                                "problem_index": p,
                                "candidate_id": row.get("candidate_id"),
                                "error": repr(exc),
                            }, indent=2),
                            encoding="utf-8",
                        )
                        raise
                    newly += 1
                if (processed <= 3 or processed % args.log_every == 0 or
                        processed == total):
                    elapsed = max(time.monotonic()-started, 1e-9)
                    rate = processed/elapsed
                    eta = (total-processed)/rate if rate > 0 else float("inf")
                    print(
                        f"[rank {args.rank}] {processed:,}/{total:,} "
                        f"({100*processed/total:.2f}%) new={newly:,} "
                        f"avg={rate:.3f} cand/s "
                        f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                        flush=True,
                    )
        writer.finish()
        if processed != total:
            raise ValueError(f"Incomplete worker: {processed}/{total}")
    finally:
        writer.close()
    print(f"[rank {args.rank}] Finished: {output}", flush=True)


def load_new_shards(config, out):
    s = config["settings"]
    n = s["scored_candidates"]
    result = {}
    for rank in (0, 1):
        for candidate_rows in groups(out / f"rank_{rank}.jsonl", n):
            p = int(candidate_rows[0]["dataset_index"])
            if p in result or p >= s["pilot_problems"] or p % 2 != rank:
                raise ValueError(f"Unexpected duplicate/new shard problem {p}")
            if not all(row.get("finite") is True for row in candidate_rows):
                raise ValueError(f"Incomplete new scoring for problem {p}")
            result[p] = candidate_rows
    if set(result) != set(range(s["pilot_problems"])):
        raise ValueError("New shards do not cover all pilot problem indices")
    return result


def assert_aligned(p, left, right, description):
    if len(left) != len(right):
        raise ValueError(f"{description}: candidate counts differ for problem {p}")
    for x, y in zip(left, right):
        if x["candidate_id"] != y["candidate_id"] or x["correct"] != y["correct"]:
            raise ValueError(f"{description}: candidate mismatch for problem {p}")
        if not math.isclose(
            finite(x["reward_model_score"]),
            finite(y["reward_model_score"]), abs_tol=1e-5, rel_tol=1e-5,
        ):
            raise ValueError(f"{description}: original RM differs for problem {p}")


def method_specs(config, old):
    specs = [{"method": "vanilla", "origin": "vanilla", "lr": 0.0, "steps": 0}]
    for step in TEST1_STEPS:
        specs.append({
            "method": f"test1_s{step}", "origin": "test1",
            "lr": 1e-6, "steps": step,
        })
    for arm in old["designs"]:
        specs.append({
            "method": f"old_{arm['id']}", "origin": "old_lhs",
            "arm": arm["id"], "lr": arm["learning_rate"],
            "steps": arm["steps"],
        })
    for i, lr in enumerate(config["lrs"]):
        for step in config["steps"]:
            specs.append({
                "method": f"new_lr{i:02d}_s{step}", "origin": "new",
                "arm": f"lr{i:02d}", "lr": lr, "steps": step,
            })
    if len({s["method"] for s in specs}) != len(specs):
        raise ValueError("Duplicate method names")
    return specs


def evaluate(args):
    config = load_config(args.output_dir)
    old_dir = Path(config["old_lhs"])
    old = read_old_design(old_dir)
    old_new, old_test1 = read_old_records(old, old_dir)
    new = load_new_shards(config, args.output_dir)
    s = config["settings"]
    specs = method_specs(config, old)
    ns = [2**i for i in range(s["scored_candidates"].bit_length())
          if 2**i <= s["scored_candidates"]]
    rng = np.random.default_rng(config["split_seed"])
    permuted = rng.permutation(s["pilot_problems"]).tolist()
    n_tune = int(round(0.6*s["pilot_problems"]))
    n_tune = min(max(1, n_tune), s["pilot_problems"]-1)
    tune = set(permuted[:n_tune])
    subsets = ("all", "tune", "holdout")
    totals = {"all": 0, "tune": 0, "holdout": 0}
    stats = {(subset, spec["method"], n): Counter()
             for subset in subsets for spec in specs for n in ns}

    for p in range(s["pilot_problems"]):
        new_rows, old_rows, test1_rows = new[p], old_new[p], old_test1[p]
        assert_aligned(p, new_rows, old_rows, "Prior LHS")
        assert_aligned(p, new_rows, test1_rows, "Test 1")
        original = np.asarray(
            [finite(row["reward_model_score"]) for row in new_rows]
        )
        truth = np.asarray([int(row["correct"]) for row in new_rows])
        scores = {"vanilla": original}
        for spec in specs[1:]:
            if spec["origin"] == "test1":
                vals = [row["scores_by_step"][str(spec["steps"])]
                        for row in test1_rows]
            elif spec["origin"] == "old_lhs":
                vals = [row["scores_by_design"][spec["arm"]]
                        for row in old_rows]
            else:
                vals = [row["scores_by_lr"][spec["arm"]][str(spec["steps"])]
                        for row in new_rows]
            scores[spec["method"]] = np.asarray([finite(v) for v in vals])
        selected_subsets = ("all", "tune") if p in tune else ("all", "holdout")
        for subset in selected_subsets:
            totals[subset] += 1
        for n in ns:
            original_winner = int(np.argmax(original[:n]))
            original_correct = int(truth[original_winner])
            for spec in specs:
                method = spec["method"]
                winner = int(np.argmax(scores[method][:n]))
                is_correct = int(truth[winner])
                for subset in selected_subsets:
                    counter = stats[(subset, method, n)]
                    counter["correct"] += is_correct
                    if winner != original_winner:
                        counter["flips"] += 1
                        if original_correct == 0 and is_correct == 1:
                            counter["I_to_C"] += 1
                        if original_correct == 1 and is_correct == 0:
                            counter["C_to_I"] += 1

    curves, summaries = [], []
    for subset in subsets:
        for spec in specs:
            method = spec["method"]
            values = []
            for n in ns:
                c = stats[(subset, method, n)]
                base = stats[(subset, "vanilla", n)]
                row = {
                    "subset": subset, "method": method,
                    "origin": spec["origin"],
                    "learning_rate": spec["lr"], "steps": spec["steps"],
                    "N": n, "num_problems": totals[subset],
                    "accuracy_percent": 100*c["correct"]/totals[subset],
                    "delta_vs_vanilla_pp":
                        100*(c["correct"]-base["correct"])/totals[subset],
                    "flip_rate_percent": 100*c["flips"]/totals[subset],
                    "I_to_C": c["I_to_C"], "C_to_I": c["C_to_I"],
                }
                curves.append(row)
                values.append(row)
            peak = max(values, key=lambda row: row["accuracy_percent"])
            last = values[-1]
            summaries.append({
                "subset": subset, "method": method,
                "origin": spec["origin"],
                "learning_rate": spec["lr"], "steps": spec["steps"],
                "peak_n": peak["N"],
                "peak_accuracy_percent": peak["accuracy_percent"],
                "final_n": last["N"],
                "final_accuracy_percent": last["accuracy_percent"],
                "final_delta_vs_vanilla_pp": last["delta_vs_vanilla_pp"],
                "final_flip_rate_percent": last["flip_rate_percent"],
                "final_I_to_C": last["I_to_C"], "final_C_to_I": last["C_to_I"],
            })

    out = args.output_dir / "evaluation"
    write_csv(out / "curves.csv", curves)
    write_csv(out / "summary.csv", summaries)
    tune_rows = sorted(
        (r for r in summaries if r["subset"] == "tune" and
         r["origin"] != "vanilla"),
        key=lambda r: (-r["final_accuracy_percent"], r["steps"],
                       r["learning_rate"], r["method"]),
    )
    chosen = []
    for rank, row in enumerate(tune_rows[:10], 1):
        h = next(r for r in summaries if r["subset"] == "holdout" and
                 r["method"] == row["method"])
        all_row = next(r for r in summaries if r["subset"] == "all" and
                       r["method"] == row["method"])
        chosen.append({
            "tune_rank": rank,
            "method": row["method"],
            "origin": row["origin"],
            "learning_rate": row["learning_rate"],
            "steps": row["steps"],
            "tune_n256_accuracy": row["final_accuracy_percent"],
            "holdout_n256_accuracy": h["final_accuracy_percent"],
            "holdout_delta_vs_vanilla_pp": h["final_delta_vs_vanilla_pp"],
            "all_n256_accuracy": all_row["final_accuracy_percent"],
        })
    write_csv(out / "top10_by_tune_n256.csv", chosen)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Small 60/40 split: show both in addition to overall outcomes.
    for subset in subsets:
        matrix = np.empty((len(config["lrs"]), len(config["steps"])))
        for i in range(len(config["lrs"])):
            for j, step in enumerate(config["steps"]):
                method = f"new_lr{i:02d}_s{step}"
                r = next(row for row in summaries
                         if row["subset"] == subset and row["method"] == method)
                matrix[i, j] = r["final_accuracy_percent"]
        fig, ax = plt.subplots(figsize=(9, 5))
        img = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(config["steps"])), config["steps"])
        ax.set_yticks(range(len(config["lrs"])),
                      [f"{lr:.2g}" for lr in config["lrs"]])
        ax.set_xlabel("SGD steps")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"New-grid BoN N=256 accuracy (%) - {subset}")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:.1f}",
                        ha="center", va="center", color="white", fontsize=10)
        fig.colorbar(img, ax=ax, label="Accuracy (%)")
        fig.tight_layout()
        fig.savefig(out / f"lr_step_heatmap_{subset}.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_methods = ["vanilla", "test1_s1", "test1_s4", "test1_s8"]
    plot_methods += [row["method"] for row in chosen[:4]]
    # Retain unique labels even if selected baseline overlaps.
    plot_methods = list(dict.fromkeys(plot_methods))
    for method in plot_methods:
        subset_rows = [
            r for r in curves
            if r["subset"] == "holdout" and r["method"] == method
        ]
        ax.plot(ns, [r["accuracy_percent"] for r in subset_rows],
                marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Best-of-N candidates")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Reward-updated BoN | top settings tuned on N=256")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "selected_holdout_accuracy_vs_n.png", dpi=200)
    plt.close(fig)

    print(f"Evaluated {len(specs)} methods, "
          f"{totals['tune']} tune and {totals['holdout']} holdout problems")
    print("N=256 ranking by tune accuracy (reward_updated ONLY):")
    for row in chosen:
        print(f"  {row['tune_rank']:>2}. {row['method']:<18} "
              f"lr={row['learning_rate']:.3g} steps={row['steps']:>2} "
              f"tune={row['tune_n256_accuracy']:5.1f}% "
              f"holdout={row['holdout_n256_accuracy']:5.1f}% "
              f"(delta {row['holdout_delta_vs_vanilla_pp']:+.1f}pp)")
    print(f"Saved results: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "worker", "evaluate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--sgd-reference", type=Path)
    parser.add_argument("--old-lhs", type=Path)
    parser.add_argument("--pilot-problems", type=int, default=100)
    parser.add_argument("--dummy-pool", type=int, default=512)
    parser.add_argument("--scored-candidates", type=int, default=256)
    parser.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--split-seed", type=int, default=1043)
    parser.add_argument(
        "--reward-model", default="OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    parser.add_argument("--reward-backend", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "init":
        init(args)
    elif args.mode == "worker":
        worker(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
