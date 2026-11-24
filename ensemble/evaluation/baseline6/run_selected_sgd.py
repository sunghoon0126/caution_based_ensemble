"""Pilot plots, select top 3 SGD configs, lambda tune top 1, full-size BoN.

prepare: rank saved pilot curves, plot every non-vanilla condition equally,
         freeze selected methods and evaluate lambda on saved pilot rewards.
worker:  score only missing selected checkpoints on one of two GPUs.
evaluate: full/previously-unseen problem BoN using updated rewards only.

Lambda score is (1-lambda)*r - lambda*(r-r_updated), with raw scores.
Full run keeps original 512-candidate fixed minimum dummy.
"""
import argparse
import csv
import json
import math
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.run_lr_step_expansion import (
    load_config, method_specs, make_adapter, write_csv,
)
from ensemble.evaluation.baseline6.run_lhs_sgd_sweep import read_design
from ensemble.evaluation.baseline6.run_steps_ablation import adapted_checkpoints

LAMBDAS = sorted(set([i/20 for i in range(21)] + [0.975, 0.99, 0.995]))


def label(spec):
    return "vanilla" if spec["method"] == "vanilla" else (
        f"step{spec['steps']}_lr{spec['lr']:.6g}"
    )


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest(out):
    return json.loads((out/"selected_config.json").read_text())


def save_figure(fig, out, name):
    import matplotlib.pyplot as plt
    fig.savefig(out/f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(pilot, specs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(pilot/"evaluation/curves.csv")
    arms = sorted([s for s in specs if s["method"] != "vanilla"],
                  key=lambda s: (s["steps"], s["lr"]))
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    patterns = ("-", "--", "-.", ":")
    for split in ("all", "tune", "holdout"):
        fig, ax = plt.subplots(figsize=(13, 8))
        for i, s in enumerate(arms):
            points = sorted([r for r in rows if r["subset"] == split
                            and r["method"] == s["method"]], key=lambda r: int(r["N"]))
            if not points:
                raise ValueError(f"Missing curve {split}/{s['method']}")
            ns = [int(r["N"]) for r in points]
            ax.plot(ns, [float(r["accuracy_percent"]) for r in points],
                    color=colors[i % len(colors)], linestyle=patterns[i % 4],
                    linewidth=1.65, alpha=.9, marker="o", markersize=3.1,
                    label=label(s))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set(xlabel="Best-of-N budget", ylabel="Accuracy (%)",
               title=f"SGD learning-rate / step sweep | {split} | {len(arms)} configurations")
        ax.grid(alpha=.18)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12),
                  ncol=4, fontsize=8, frameon=False)
        fig.tight_layout()
        save_figure(fig, out, f"all_sgd_equal_{split}")


class Source:
    """Stream complete cached groups in increasing dataset_index order."""
    def __init__(self, path, n):
        self.iterator = iter(groups(Path(path), n))
        self.current = None
        self.exhausted = False

    def get(self, p):
        while not self.exhausted and (self.current is None
                                      or int(self.current[0]["dataset_index"]) < p):
            self.current = next(self.iterator, None)
            if self.current is None:
                self.exhausted = True
        if self.current is not None and int(self.current[0]["dataset_index"]) == p:
            return self.current
        return None

    def close(self):
        self.iterator.close()


def open_sources(manifest, rank):
    pilot = Path(manifest["pilot"])
    cfg = manifest["pilot_config"]
    s = cfg["settings"]
    n = s["scored_candidates"]
    origins = {a["origin"] for a in manifest["selected"]}
    sources = {}
    if "new" in origins:
        sources["new"] = Source(pilot/f"rank_{rank}.jsonl", n)
    if "old_lhs" in origins:
        sources["old_lhs"] = Source(Path(cfg["old_lhs"])/f"rank_{rank}.jsonl", n)
    if "test1" in origins:
        sources["test1"] = Source(s["sgd_reference"], n)
    return sources


def verify_cached(row, cached, fingerprint, dummy):
    if cached["candidate_id"] != row["candidate_id"] or cached["correct"] != row["correct"]:
        raise ValueError("Cached candidate mismatch")
    if not cached.get("finite") or cached.get("input_fingerprint") != fingerprint:
        raise ValueError(f"Cached input/config provenance mismatch: {row['candidate_id']}")
    if not math.isclose(finite(cached["dummy_reward"]), dummy, abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError("Cached dummy mismatch")
    if not math.isclose(finite(cached["reward_model_score"]),
                        finite(row["reward_model_score"]), abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError("Cached original score mismatch")


def cached_score(spec, row):
    if spec["origin"] == "test1":
        return finite(row["scores_by_step"][str(spec["steps"])])
    if spec["origin"] == "old_lhs":
        return finite(row["scores_by_design"][spec["arm"]])
    if spec["origin"] == "new":
        return finite(row["scores_by_lr"][spec["arm"]][str(spec["steps"])])
    raise ValueError("Unknown source")


def get_cached(manifest, sources, p):
    return {origin: src.get(p) for origin, src in sources.items()
            if origin == "test1" or p < manifest["pilot_config"]["settings"]["pilot_problems"]}


def summarize(counts, totals, methods, ns):
    curves = []
    for subset, num in totals.items():
        if not num:
            continue
        for method in methods:
            for n in ns:
                c = counts[subset, method, n]
                b = counts[subset, methods[0], n]
                curves.append({"subset": subset, "method": method, "N": n,
                    "num_problems": num, "accuracy_percent": 100*c["correct"]/num,
                    "delta_vs_vanilla_pp": 100*(c["correct"]-b["correct"])/num,
                    "flip_rate_percent": 100*c["flip"]/num,
                    "I_to_C": c["I_to_C"], "C_to_I": c["C_to_I"]})
    return curves


def add_problem(counts, totals, subsets, truth, scores, ns):
    names = list(scores)
    for subset in subsets:
        totals[subset] += 1
    for n in ns:
        base = int(np.argmax(scores[names[0]][:n]))
        for method, score in scores.items():
            winner = int(np.argmax(score[:n]))
            for subset in subsets:
                c = counts[subset, method, n]
                c["correct"] += int(truth[winner])
                c["flip"] += int(winner != base)
                c["I_to_C"] += int(not truth[base] and truth[winner])
                c["C_to_I"] += int(truth[base] and not truth[winner])


def lambda_pilot(manifest, out):
    cfg = manifest["pilot_config"]
    s = cfg["settings"]
    best = manifest["selected"][0]
    # Only open sources necessary for the top-1 arm.
    subset_manifest = dict(manifest, selected=[best])
    sources = {rank: open_sources(subset_manifest, rank) for rank in (0, 1)}
    num = s["pilot_problems"]
    nmax = s["scored_candidates"]
    ns = [2**i for i in range(nmax.bit_length())]
    perm = np.random.default_rng(cfg["split_seed"]).permutation(num).tolist()
    tune = set(perm[:min(max(1, int(round(.6*num))), num-1)])
    counts, totals = defaultdict(Counter), Counter()
    names = [f"{x:g}" for x in LAMBDAS]
    try:
        for p, full in enumerate(groups(s["original"], s["dummy_pool"])):
            if p >= num:
                break
            if int(full[0]["dataset_index"]) != p:
                raise ValueError("Expected zero-based sequential dataset index")
            cache = get_cached(subset_manifest, sources[p%2], p).get(best["origin"])
            if cache is None:
                raise ValueError(f"Missing top-1 cached scores for problem {p}")
            dummy = min(finite(r["reward_model_score"]) for r in full)
            context = {"dummy_reward": dummy, "pool_hash": digest(full)}
            original, updated, truth = [], [], []
            for i, row in enumerate(full[:nmax]):
                verify_cached(row, cache[i], digest({"source": row, "context": context}), dummy)
                original.append(finite(row["reward_model_score"]))
                updated.append(cached_score(best, cache[i]))
                truth.append(int(row["correct"]))
            original, updated = np.asarray(original), np.asarray(updated)
            penalty = original-updated
            # Use exactly updated/2 at lambda=.5 to avoid cancellation changing ties.
            scores = {name: (.5*updated if lam == .5 else
                        (1-lam)*original-lam*penalty)
                      for name, lam in zip(names, LAMBDAS)}
            add_problem(counts, totals, ("all", "tune" if p in tune else "holdout"),
                        truth, scores, ns)
    finally:
        for srcs in sources.values():
            for src in srcs.values():
                src.close()
    if totals["all"] != num:
        raise ValueError("Incomplete pilot")
    curves = summarize(counts, totals, names, ns)
    for r in curves:
        r["lambda"] = float(r.pop("method"))
    write_csv(out/"lambda_curves.csv", curves)
    finals = [r for r in curves if r["N"] == nmax]
    write_csv(out/"lambda_n256.csv", finals)
    tuned = sorted([r for r in finals if r["subset"] == "tune"],
                   key=lambda r: (-r["accuracy_percent"], r["lambda"]))
    best_lambda = tuned[0]["lambda"]
    check = next(r for r in finals if r["subset"] == "holdout" and r["lambda"] == best_lambda)
    (out/"selected_lambda.json").write_text(json.dumps({
        "setting": best, "selected_lambda": best_lambda, "tune": tuned[0],
        "holdout": check, "formula": "(1-lambda)*r-lambda*(r-r_updated)"
    }, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in ("all", "tune", "holdout"):
        pts = sorted([r for r in finals if r["subset"] == split], key=lambda r:r["lambda"])
        ax.plot([r["lambda"] for r in pts], [r["accuracy_percent"] for r in pts],
                marker="o", markersize=3, label=split)
    ax.axvline(.5, linestyle=":", color="gray", label="updated reward equivalent")
    ax.axvline(best_lambda, linestyle="--", color="gray", alpha=.65,
               label=f"tune winner: {best_lambda:g}")
    ax.set(xlabel="Lambda", ylabel=f"N={nmax} accuracy (%)", title=f"Lambda tuning | {label(best)}")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    save_figure(fig, out, "lambda_accuracy_n256")
    fig, ax = plt.subplots(figsize=(9, 5))
    for lam in sorted({0., .5, best_lambda, 1.}):
        pts = [r for r in curves if r["subset"] == "holdout" and r["lambda"] == lam]
        ax.plot(ns, [r["accuracy_percent"] for r in pts], marker="o", label=f"lambda={lam:g}")
    ax.set_xscale("log", base=2); ax.set_xticks(ns, [str(n) for n in ns])
    ax.set(xlabel="Best-of-N budget", ylabel="Holdout accuracy (%)", title=f"Lambda tuning | {label(best)}")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    save_figure(fig, out, "lambda_holdout_accuracy_vs_n")
    print(f"Best lambda: {best_lambda:g}; tune={tuned[0]['accuracy_percent']:.2f}%; "
          f"holdout={check['accuracy_percent']:.2f}%", flush=True)


def prepare(args):
    pilot = args.pilot.resolve()
    cfg = load_config(pilot)
    old = read_design(Path(cfg["old_lhs"]))
    specs = method_specs(cfg, old)
    lookup = {s["method"]: s for s in specs}
    s = cfg["settings"]
    if s["scored_candidates"] != 256 or args.full_candidates != s["dummy_pool"]:
        raise ValueError("Expected pilot N=256 and full N=dummy pool")
    if args.full_problems < s["pilot_problems"]:
        raise ValueError("Full problem count must include pilot")
    summaries = read_csv(pilot/"evaluation/summary.csv")
    rank = sorted([r for r in summaries if r["subset"] == args.select_split
                   and r["method"] != "vanilla"],
                  key=lambda r: (-float(r["final_accuracy_percent"]), int(r["steps"]),
                                 float(r["learning_rate"]), r["method"]))
    selected, seen = [], set()
    for r in rank:
        spec = lookup[r["method"]]
        key = spec["lr"], spec["steps"]
        if key in seen:
            continue
        if int(r["final_n"]) != 256 or key != (float(r["learning_rate"]), int(r["steps"])):
            raise ValueError("Summary/design mismatch")
        selected.append(dict(spec, pilot_accuracy=float(r["final_accuracy_percent"])))
        seen.add(key)
        if len(selected) == args.top_k:
            break
    if len(selected) != args.top_k:
        raise ValueError(f"Need {args.top_k} distinct settings")
    manifest = {"version": 1, "pilot": str(pilot), "pilot_config": cfg,
                "selected": selected, "select_split": args.select_split,
                "full_problems": args.full_problems, "full_candidates": args.full_candidates}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    f = out/"selected_config.json"
    if f.exists() and json.loads(f.read_text()) != manifest:
        raise ValueError("Selected config changed; use a fresh output directory")
    f.write_text(json.dumps(manifest, indent=2))
    write_csv(out/f"selected_top{args.top_k}.csv", [
        {"rank": i+1, "method": label(a), "source": a["method"],
         "learning_rate": a["lr"], "steps": a["steps"],
         "selection_split": args.select_split, "pilot_n256_accuracy": a["pilot_accuracy"]}
        for i, a in enumerate(selected)])
    for a in selected:
        print(f"Selected {label(a)} ({a['method']}): {a['pilot_accuracy']:.2f}%", flush=True)
    plot_all(pilot, specs, out)
    if not args.skip_lambda:
        lambda_pilot(manifest, out)
    print(f"Preparation complete: {out}", flush=True)


def worker(args):
    m = load_manifest(args.output_dir)
    s = m["pilot_config"]["settings"]
    arms = m["selected"]
    n, problems = m["full_candidates"], m["full_problems"]
    if args.rank not in (0, 1) or args.log_every <= 0:
        raise ValueError("Invalid rank/log interval")
    sources = open_sources(m, args.rank)
    writer = ResumeWriter(args.output_dir/f"rank_{args.rank}.jsonl",
                          {"manifest": m, "rank": args.rank})
    total = sum(p%2 == args.rank for p in range(problems))*n
    done = reused = calculated = 0
    adapter = None
    start = time.monotonic()
    print(f"[rank {args.rank}] {total:,} candidates; {[label(a) for a in arms]}", flush=True)
    try:
        for p, full in enumerate(groups(s["original"], s["dummy_pool"])):
            if p >= problems:
                break
            if int(full[0]["dataset_index"]) != p:
                raise ValueError("Expected zero-based sequential dataset index")
            if p%2 != args.rank:
                continue
            cache = get_cached(m, sources, p)
            dummy = min(finite(row["reward_model_score"]) for row in full)
            context = {"dummy_reward": dummy, "pool_hash": digest(full)}
            for i, row in enumerate(full[:n]):
                completed, prov = writer.completed(row, context)
                if not completed:
                    scores = {}
                    missing = defaultdict(list)
                    for a in arms:
                        cached = cache.get(a["origin"])
                        if cached is not None and i < len(cached):
                            verify_cached(row, cached[i], prov["input_fingerprint"], dummy)
                            scores[a["method"]] = cached_score(a, cached[i])
                            reused += 1
                        else:
                            missing[a["lr"]].append(a)
                    if missing:
                        if adapter is None:
                            print(f"[rank {args.rank}] Loading RM", flush=True)
                            adapter = make_adapter(s, args.device, row)
                            print(f"[rank {args.rank}] RM ready", flush=True)
                        inputs = adapter.inputs(row["prompt"], row["response"])
                        for lr, trajectory in missing.items():
                            steps = tuple(sorted({a["steps"] for a in trajectory}))
                            before, checkpoints = adapted_checkpoints(adapter.model, inputs, dummy, lr, steps)
                            if not math.isclose(before, finite(row["reward_model_score"]),
                                                abs_tol=1e-4, rel_tol=1e-4):
                                raise ValueError("Original RM score mismatch")
                            for a in trajectory:
                                scores[a["method"]] = finite(checkpoints[str(a["steps"])])
                                calculated += 1
                    compact = {key: row[key] for key in (
                        "dataset", "split", "trial_id", "dataset_index", "instance_id",
                        "sample_index", "candidate_id", "correct", "reward_model_score")}
                    writer.write(dict(compact, **prov, dummy_reward=dummy,
                                      scores_by_method=scores,
                                      reward_after=scores[arms[0]["method"]], finite=True))
                done += 1
                if done <= 3 or done%args.log_every == 0 or done == total:
                    elapsed = max(time.monotonic()-start, 1e-9)
                    print(f"[rank {args.rank}] {done:,}/{total:,} ({100*done/total:.2f}%) "
                          f"reused scores={reused:,} new scores={calculated:,}; "
                          f"avg={done/elapsed:.3f} cand/s; "
                          f"ETA={(total-done)*elapsed/done/60:.1f} min", flush=True)
        writer.finish()
        if done != total:
            raise ValueError(f"Missing candidates: {done}/{total}")
    finally:
        writer.close()
        for src in sources.values():
            src.close()
    print(f"[rank {args.rank}] Complete", flush=True)


def evaluate(args):
    m = load_manifest(args.output_dir)
    methods = ["vanilla"] + [a["method"] for a in m["selected"]]
    n = m["full_candidates"]
    ns = [2**i for i in range(n.bit_length()) if 2**i <= n]
    counts, totals = defaultdict(Counter), Counter()
    pilot_count = m["pilot_config"]["settings"]["pilot_problems"]
    seen = set()
    for rank in (0, 1):
        for rows in groups(args.output_dir/f"rank_{rank}.jsonl", n):
            p = int(rows[0]["dataset_index"])
            if p in seen or p%2 != rank or p >= m["full_problems"]:
                raise ValueError(f"Unexpected problem {p}")
            seen.add(p)
            if any(not r.get("finite") or r["backprop_config"] != {"manifest":m,"rank":rank} for r in rows):
                raise ValueError("Invalid result")
            original = np.asarray([finite(r["reward_model_score"]) for r in rows])
            scores = {"vanilla": original}
            for a in m["selected"]:
                scores[a["method"]] = np.asarray([finite(r["scores_by_method"][a["method"]]) for r in rows])
            add_problem(counts, totals, ("all", "pilot" if p < pilot_count else "unseen"),
                        [int(r["correct"]) for r in rows], scores, ns)
    if seen != set(range(m["full_problems"])):
        raise ValueError("Incomplete full experiment")
    curves = summarize(counts, totals, methods, ns)
    labels = {"vanilla":"vanilla", **{a["method"]:label(a) for a in m["selected"]}}
    for r in curves:
        r["setting"] = labels[r["method"]]
    out = args.output_dir/"evaluation"
    out.mkdir(exist_ok=True)
    write_csv(out/"curves.csv", curves)
    summary = []
    for split in totals:
        for method in methods:
            pts = [r for r in curves if r["subset"] == split and r["method"] == method]
            peak = max(pts, key=lambda r:r["accuracy_percent"])
            final = pts[-1]
            summary.append({"subset":split,"setting":labels[method],"num_problems":totals[split],
                            "peak_n":peak["N"],"peak_accuracy":peak["accuracy_percent"],
                            "final_n":n,"final_accuracy":final["accuracy_percent"],
                            "final_delta_pp":final["delta_vs_vanilla_pp"],
                            "degradation_pp":peak["accuracy_percent"]-final["accuracy_percent"]})
    write_csv(out/"summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split in ("all","unseen"):
        if not totals[split]:
            continue
        fig, ax = plt.subplots(figsize=(9,5.5))
        for method in methods:
            pts = [r for r in curves if r["subset"]==split and r["method"]==method]
            ax.plot(ns,[r["accuracy_percent"] for r in pts],marker="o",linewidth=1.8,label=labels[method])
        ax.set_xscale("log",base=2); ax.set_xticks(ns,[str(v) for v in ns])
        ax.set(xlabel="Best-of-N budget",ylabel="Accuracy (%)",title=f"Full SGD evaluation | {split} ({totals[split]} problems)")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        save_figure(fig,out,f"accuracy_vs_n_{split}")
    for row in summary:
        print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare","worker","evaluate","lambda"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pilot",type=Path)
    p.add_argument("--select-split",choices=("tune","all"),default="tune")
    p.add_argument("--top-k",type=int,choices=(1,2,3),default=3)
    p.add_argument("--skip-lambda",action="store_true")
    p.add_argument("--full-problems",type=int,default=1000)
    p.add_argument("--full-candidates",type=int,default=512)
    p.add_argument("--rank",type=int,default=-1)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--log-every",type=int,default=20)
    args = p.parse_args()
    if args.mode=="prepare":
        if args.pilot is None:
            p.error("prepare requires --pilot")
        prepare(args)
    elif args.mode=="worker":
        worker(args)
    elif args.mode=="lambda":
        lambda_pilot(load_manifest(args.output_dir), args.output_dir)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
