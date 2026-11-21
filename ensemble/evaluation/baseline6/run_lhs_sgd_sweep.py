"""Baseline 6: two-GPU sharded Latin-hypercube SGD (LR, steps) screen.

Modes
-----
init:     write immutable randomized-LHS design (no GPU)
worker:   score assigned problem subset using one visible GPU, resumable JSONL
evaluate: combine rank 0/1 scores and run BoN + post-hoc lambda sweep (no GPU)

Candidate-specific full RM SGD is stateless; the original reward model and
fixed per-prompt minimum dummy over ALL 512 originals are unchanged.
Each sampled (LR, number of steps) uses an independent SGD trajectory, and
each candidate starts from original theta0. Baseline 1e-6 scores are reused
from Test 1, not recomputed. Score lambda = r0-lambda*(r0-r_updated).
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.reward_model_adapter import RewardModelAdapter
from ensemble.evaluation.baseline6.run_steps_ablation import adapted_checkpoints


DEFAULT_LAMBDAS = (0.5, 1.0, 2.0, 4.0, 8.0, 20.0)
BASELINE_STEPS = (1, 2, 4, 8)


def read_design(directory):
    path = directory / "lhs_design.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("Unsupported/missing design manifest")
    if len(data["designs"]) != data["settings"]["num_designs"]:
        raise ValueError("Design count mismatch")
    return data


def initialize(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "original": str(args.original.resolve()),
        "sgd_reference": str(args.sgd_reference.resolve()),
        "pilot_problems": args.pilot_problems,
        "dummy_pool": args.dummy_pool,
        "scored_candidates": args.scored_candidates,
        "num_designs": args.num_designs,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "steps": list(args.steps),
        "seed": args.seed,
        "reward_model": args.reward_model,
        "reward_backend": args.reward_backend,
        "dtype": args.dtype,
        "revision": args.revision,
        "max_length": args.max_length,
    }
    if not (args.pilot_problems > 0 and args.dummy_pool >= args.scored_candidates > 0
            and args.scored_candidates & (args.scored_candidates-1) == 0
            and args.num_designs > 0 and args.min_lr > 0 and args.max_lr >= args.min_lr
            and args.max_lr <= 1.0e-6 and all(k > 0 for k in args.steps)
            and args.steps == sorted(set(args.steps))):
        raise ValueError("Invalid design range or sample sizes")
    rng = np.random.default_rng(args.seed)
    n = args.num_designs
    lhs_lr = (rng.permutation(n) + rng.random(n)) / n
    lhs_step = (rng.permutation(n) + rng.random(n)) / n
    designs = []
    for i in range(n):
        lr = float(np.exp(np.log(args.min_lr) +
                          lhs_lr[i] * (np.log(args.max_lr) - np.log(args.min_lr))))
        index = min(int(lhs_step[i] * len(args.steps)), len(args.steps) - 1)
        designs.append({
            "id": f"lhs_{i:02d}",
            "learning_rate": lr,
            "steps": args.steps[index],
            "lhs_lr_coordinate": float(lhs_lr[i]),
            "lhs_step_coordinate": float(lhs_step[i]),
        })
    designs.sort(key=lambda row: (row["learning_rate"], row["steps"]))
    for i, row in enumerate(designs):
        row["id"] = f"lhs_{i:02d}"
    data = {"version": 1, "settings": settings, "designs": designs}
    path = out / "lhs_design.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError("Design already exists and differs; choose another output dir")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with (out / "lhs_design.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(designs[0]))
        writer.writeheader()
        writer.writerows(designs)
    print(f"Saved {n} LHS (LR, steps) designs in {path}")
    print("ID      Learning rate    Steps   LR*Steps")
    for d in designs:
        print(f"{d['id']:<8}{d['learning_rate']:<17.5g}{d['steps']:<8d}"
              f"{d['learning_rate']*d['steps']:.5g}")
    print(f"Total SGD gradient steps per candidate: {sum(d['steps'] for d in designs)}")


def make_adapter(settings, device, row):
    adapter = RewardModelAdapter(
        settings["reward_model"], device, settings["dtype"],
        settings["reward_backend"], settings["revision"], settings["max_length"],
    )
    if row.get("reward_model_path") != settings["reward_model"]:
        raise ValueError("Wrong original reward model")
    if row.get("reward_scoring_config") != adapter.settings:
        raise ValueError("Original RM tokenization/precision settings differ")
    recorded = row.get("reward_model_revision")
    resolved = getattr(adapter.model.config, "_commit_hash", None)
    if recorded and resolved != recorded:
        raise ValueError("Original RM revision mismatch")
    return adapter


def run_worker(args):
    design = read_design(args.output_dir)
    s = design["settings"]
    if args.world_size != 2 or args.rank not in (0, 1):
        raise ValueError("This launch expects exactly two ranks, 0 and 1")
    local_path = args.output_dir / f"rank_{args.rank}.jsonl"
    run_config = {
        "version": design["version"],
        "settings": s,
        "designs": design["designs"],
        "shard_rank": args.rank,
        "world_size": args.world_size,
    }
    writer = ResumeWriter(local_path, run_config)
    adapter = None
    own_problems = sum(p % args.world_size == args.rank
                       for p in range(s["pilot_problems"]))
    target_count = own_problems * s["scored_candidates"]
    seen = newly = 0
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=target_count, desc=f"SGD LHS GPU rank {args.rank}",
                   unit="cand", dynamic_ncols=True)
    except ImportError:
        bar = None
    try:
        for p, full in enumerate(groups(Path(s["original"]), s["dummy_pool"])):
            if p >= s["pilot_problems"]:
                break
            if p % args.world_size != args.rank:
                continue
            minimum = min(finite(row["reward_model_score"]) for row in full)
            context = {"dummy_reward": minimum, "pool_hash": digest(full)}
            for row in full[:s["scored_candidates"]]:
                completed, provenance = writer.completed(row, context)
                seen += 1
                if completed:
                    if bar:
                        bar.update(1)
                    continue
                if adapter is None:
                    print(f"Loading RM on {args.device} for rank {args.rank}", flush=True)
                    adapter = make_adapter(s, args.device, row)
                    print("Reward model ready", flush=True)
                try:
                    inputs = adapter.inputs(row["prompt"], row["response"])
                    results = {}
                    before = None
                    for arm in design["designs"]:
                        current_before, score_map = adapted_checkpoints(
                            adapter.model, inputs, minimum,
                            arm["learning_rate"], (arm["steps"],),
                        )
                        if before is None:
                            before = current_before
                        elif not math.isclose(before, current_before,
                                              rel_tol=1e-5, abs_tol=1e-5):
                            raise ValueError("Different original reward between design arms")
                        results[arm["id"]] = finite(score_map[str(arm["steps"])])
                    original = finite(row["reward_model_score"])
                    if not math.isclose(before, original, rel_tol=1e-4, abs_tol=1e-4):
                        raise ValueError("Original model reward differs from saved score")
                    writer.write({
                        **row, **provenance,
                        "dummy_reward": minimum,
                        "reward_before": before,
                        "reward_after": results[design["designs"][-1]["id"]],
                        "scores_by_design": results,
                        "finite": True,
                    })
                except Exception as exc:
                    (args.output_dir / f"failure_rank_{args.rank}.json").write_text(
                        json.dumps({"problem_idx": p, "candidate_id": row.get("candidate_id"),
                                    "exception": repr(exc)}, indent=2),
                        encoding="utf-8",
                    )
                    raise
                newly += 1
                if bar:
                    bar.update(1)
            if p == args.rank or (p + 1) % 10 == 0:
                print(f"[rank {args.rank}] visited {seen:,}/{target_count:,}, "
                      f"new {newly:,}, problem index={p}", flush=True)
        writer.finish()
        if seen != target_count:
            raise ValueError(f"Rank {args.rank}: expected {target_count}, saw {seen}")
    finally:
        writer.close()
        if bar:
            bar.close()
    print(f"Rank {args.rank} complete: {local_path}", flush=True)


def load_records(design, out):
    s = design["settings"]
    n = s["scored_candidates"]
    result = {}
    for rank in (0, 1):
        path = out / f"rank_{rank}.jsonl"
        for rows in groups(path, n):
            index = int(rows[0]["dataset_index"])
            if index in result or index >= s["pilot_problems"] or index % 2 != rank:
                raise ValueError(f"Invalid/duplicate shard problem index {index}")
            if any(row.get("finite") is not True for row in rows):
                raise ValueError(f"Unfinished scoring in shard {rank}, problem {index}")
            result[index] = rows
    if len(result) != s["pilot_problems"]:
        raise ValueError(f"Expected {s['pilot_problems']} distinct problems, found {len(result)}")

    reference = {}
    for idx, rows in enumerate(groups(Path(s["sgd_reference"]), n)):
        if idx >= s["pilot_problems"]:
            break
        reference[int(rows[0]["dataset_index"])] = rows
    if set(reference) != set(result):
        raise ValueError("Reference SGD and LHS problem IDs disagree")
    for idx, rows in result.items():
        ref_rows = reference[idx]
        for a, b in zip(rows, ref_rows):
            if a["candidate_id"] != b["candidate_id"]:
                raise ValueError(f"Reference candidate mismatch at problem {idx}")
            if a["correct"] != b["correct"]:
                raise ValueError(f"Reference ground truth mismatch at problem {idx}")
            if not math.isclose(finite(a["reward_model_score"]),
                                finite(b["reward_model_score"]),
                                rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Reference original score mismatch at problem {idx}")
    return result, reference


def fmt_lambda(value):
    return format(float(value), "g").replace(".", "p")


def all_methods(design, lambdas):
    result = [("vanilla", None, None)]
    result.extend((f"sgd_1e-6_step{k}", f"old_{k}", 1.0)
                  for k in BASELINE_STEPS)
    for arm in design["designs"]:
        for lam in lambdas:
            result.append((f"{arm['id']}_lam{fmt_lambda(lam)}", arm["id"], lam))
    return result


def evaluate(args):
    design = read_design(args.output_dir)
    s = design["settings"]
    rows_by_idx, reference_by_idx = load_records(design, args.output_dir)
    lambdas = list(args.lambdas)
    if (not lambdas or len(set(lambdas)) != len(lambdas)
            or any(not math.isfinite(v) or v <= 0 for v in lambdas)):
        raise ValueError("Lambda values must be distinct, positive and finite")
    methods = all_methods(design, lambdas)
    method_names = [m for m, _, _ in methods]
    n = s["scored_candidates"]
    ns = [2 ** i for i in range(n.bit_length()) if 2 ** i <= n]

    rng = np.random.default_rng(s["seed"] + 1001)
    shuffled = rng.permutation(s["pilot_problems"]).tolist()
    tune_count = max(1, int(0.6 * s["pilot_problems"]))
    if tune_count >= s["pilot_problems"]:
        tune_count = s["pilot_problems"] - 1
    if tune_count == 0:
        raise ValueError("Pilot requires at least two problems for tune/holdout split")
    tune = set(shuffled[:tune_count])
    subsets = ("all", "tune", "holdout")
    counts = {(subset, method, size): Counter()
              for subset in subsets for method in method_names for size in ns}
    totals = {"all": 0, "tune": 0, "holdout": 0}

    for pos in range(s["pilot_problems"]):
        own = rows_by_idx[pos]
        reference = reference_by_idx[pos]
        original = np.asarray([finite(row["reward_model_score"]) for row in own])
        truth = np.asarray([int(row["correct"]) for row in own], dtype=int)
        raw = {}
        for k in BASELINE_STEPS:
            raw[f"old_{k}"] = np.asarray(
                [finite(row["scores_by_step"][str(k)]) for row in reference]
            )
        for arm in design["designs"]:
            raw[arm["id"]] = np.asarray(
                [finite(row["scores_by_design"][arm["id"]]) for row in own]
            )
        subset_names = ("all", "tune") if pos in tune else ("all", "holdout")
        for subset in subset_names:
            totals[subset] += 1
        for size in ns:
            vanilla_winner = int(np.argmax(original[:size]))
            vanilla_correct = int(truth[vanilla_winner])
            for method, raw_id, lam in methods:
                if raw_id is None:
                    winner = vanilla_winner
                else:
                    score = original if lam == 0 else original - lam*(original - raw[raw_id])
                    winner = int(np.argmax(score[:size]))
                correct = int(truth[winner])
                for subset in subset_names:
                    c = counts[(subset, method, size)]
                    c["correct"] += correct
                    if winner != vanilla_winner:
                        c["flips"] += 1
                        if vanilla_correct == 0 and correct == 1:
                            c["I_to_C"] += 1
                        elif vanilla_correct == 1 and correct == 0:
                            c["C_to_I"] += 1
    curves = []
    for subset in subsets:
        num = totals[subset]
        if num <= 0:
            raise ValueError(f"Empty subset {subset}")
        for method in method_names:
            for size in ns:
                c = counts[(subset, method, size)]
                base = counts[(subset, "vanilla", size)]
                curves.append({
                    "subset": subset,
                    "method": method,
                    "N": size,
                    "num_problems": num,
                    "accuracy_percent": 100*c["correct"]/num,
                    "delta_vs_vanilla_pp": 100*(c["correct"]-base["correct"])/num,
                    "flip_rate_percent": 100*c["flips"]/num,
                    "I_to_C": c["I_to_C"],
                    "C_to_I": c["C_to_I"],
                })
    metrics = []
    for subset in subsets:
        for method, raw_id, lam in methods:
            series = [r for r in curves if r["subset"] == subset and r["method"] == method]
            peak = max(series, key=lambda row: row["accuracy_percent"])
            end = series[-1]
            arm = next((a for a in design["designs"] if a["id"] == raw_id), None)
            metrics.append({
                "subset": subset, "method": method,
                "lr": arm["learning_rate"] if arm else (1e-6 if raw_id else 0),
                "steps": arm["steps"] if arm else (
                    int(raw_id.split("_")[1]) if raw_id else 0
                ),
                "lambda": lam if lam is not None else 0,
                "peak_n": peak["N"],
                "peak_accuracy_percent": peak["accuracy_percent"],
                "final_n": end["N"],
                "final_accuracy_percent": end["accuracy_percent"],
                "final_delta_pp": end["delta_vs_vanilla_pp"],
                "final_flip_rate_percent": end["flip_rate_percent"],
                "final_I_to_C": end["I_to_C"],
                "final_C_to_I": end["C_to_I"],
            })

    out = args.output_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in (("curves.csv", curves), ("summary.csv", metrics)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    tune_rows = [row for row in metrics
                 if row["subset"] == "tune" and row["method"].startswith("lhs_")]
    tune_rows.sort(key=lambda r: (-r["final_accuracy_percent"], r["steps"],
                                  r["lambda"], r["lr"]))
    top_ids = [item["method"] for item in tune_rows[:5]]
    with (out / "top5_by_tune_n256.csv").open("w", newline="", encoding="utf-8") as handle:
        items = []
        for rank, item in enumerate(tune_rows[:5], start=1):
            holdout = next(r for r in metrics if r["subset"] == "holdout"
                           and r["method"] == item["method"])
            items.append({
                "rank_on_tune": rank, "method": item["method"],
                "learning_rate": item["lr"], "steps": item["steps"],
                "lambda": item["lambda"], "tune_n256_accuracy": item["final_accuracy_percent"],
                "holdout_n256_accuracy": holdout["final_accuracy_percent"],
                "holdout_n256_delta_vs_vanilla_pp": holdout["final_delta_pp"],
            })
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in ["vanilla", "sgd_1e-6_step1", "sgd_1e-6_step4", "sgd_1e-6_step8"] + top_ids[:3]:
        series = [r for r in curves if r["subset"] == "holdout" and r["method"] == method]
        ax.plot(ns, [r["accuracy_percent"] for r in series], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(k) for k in ns])
    ax.set(xlabel="Best-of-N budget", ylabel="Holdout accuracy (%)",
           title="SGD LHS sweep - top 3 by tune N=256 (pilot holdout)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "top3_holdout_accuracy_vs_n.png", dpi=200)
    plt.close(fig)
    print(f"Evaluated {s['pilot_problems']} problems; "
          f"tune={totals['tune']}, holdout={totals['holdout']}", flush=True)
    print("Top 5 tuned on N=256, tested on disjoint pilot holdout:")
    for item in items:
        print(item)
    print(f"Saved: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("init", "worker", "evaluate"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--original", type=Path)
    p.add_argument("--sgd-reference", type=Path)
    p.add_argument("--pilot-problems", type=int, default=100)
    p.add_argument("--scored-candidates", type=int, default=256)
    p.add_argument("--dummy-pool", type=int, default=512)
    p.add_argument("--num-designs", type=int, default=10)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--max-lr", type=float, default=1e-6)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-model", default="OpenAssistant/reward-model-deberta-v3-large-v2")
    p.add_argument("--reward-backend", default="auto")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--revision", default="main")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--rank", type=int, default=-1)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    args = p.parse_args()
    if args.mode == "init":
        if args.original is None or args.sgd_reference is None:
            p.error("init requires --original and --sgd-reference")
        initialize(args)
    elif args.mode == "worker":
        run_worker(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
