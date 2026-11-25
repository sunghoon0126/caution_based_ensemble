"""Plot signed reward penalties for all saved SGD LR/step pilot arms.

No model loading or GPU scoring. Use the same first 100 x 256 candidates as
the pilot, including cached Test 1 and LHS arms. For each arm report candidate-
weighted mean original-minus-updated reward separately for correct/incorrect.
Also report the same analysis for per-problem top-16 original-RM candidates.
The LR axis is categorical, sorted numerically; untested cells are blank.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from ensemble.evaluation.baseline6.common import groups, finite, digest
from ensemble.evaluation.baseline6.run_lr_step_expansion import (
    load_config, method_specs, write_csv,
)
from ensemble.evaluation.baseline6.run_lhs_sgd_sweep import read_design
from ensemble.evaluation.baseline6.run_selected_sgd import (
    open_sources, get_cached, verify_cached, cached_score, label,
)


def calculate(pilot, config, specs, top_k):
    s = config["settings"]
    count = s["scored_candidates"]
    if not 1 <= top_k <= count:
        raise ValueError("top-k must be in 1..scored_candidates")
    manifest = {"pilot": str(pilot), "pilot_config": config,
                "selected": specs}
    sources = {r: open_sources(manifest, r) for r in (0, 1)}
    data = defaultdict(lambda: [[], []])
    seen = 0
    try:
        for p, full in enumerate(groups(Path(s["original"]), s["dummy_pool"])):
            if p >= s["pilot_problems"]:
                break
            if int(full[0]["dataset_index"]) != p:
                raise ValueError("Expected consecutive dataset indices")
            caches = get_cached(manifest, sources[p % 2], p)
            if any(caches.get(a["origin"]) is None for a in specs):
                raise ValueError(f"Missing cached group at problem {p}")
            original_rows = full[:count]
            dummy = min(finite(row["reward_model_score"]) for row in full)
            context = {"dummy_reward": dummy, "pool_hash": digest(full)}
            orig = np.array([finite(row["reward_model_score"]) for row in original_rows])
            truth = np.array([bool(row["correct"]) for row in original_rows])
            for i, row in enumerate(original_rows):
                fingerprint = digest({"source": row, "context": context})
                for cached in caches.values():
                    verify_cached(row, cached[i], fingerprint, dummy)
            # Sort the ORIGINAL scores once; same top-k population for all arms.
            top = np.argsort(-orig, kind="stable")[:top_k]
            populations = {"all": np.arange(count), f"top{top_k}_original": top}
            for a in specs:
                updated = np.array([cached_score(a, row) for row in caches[a["origin"]]])
                penalty = orig - updated
                for scope, ids in populations.items():
                    good = penalty[ids[truth[ids]]]
                    bad = penalty[ids[~truth[ids]]]
                    data[scope, a["method"]][0].append(good)
                    data[scope, a["method"]][1].append(bad)
            seen += 1
            if seen == 1 or seen % 10 == 0 or seen == s["pilot_problems"]:
                print(f"Loaded {seen}/{s['pilot_problems']} pilot problems", flush=True)
    finally:
        for group in sources.values():
            for src in group.values():
                src.close()
    if seen != s["pilot_problems"]:
        raise ValueError("Incomplete pilot")
    rows = []
    for (scope, method), (good, bad) in data.items():
        a = next(a for a in specs if a["method"] == method)
        c, w = np.concatenate(good), np.concatenate(bad)
        if not len(c) or not len(w):
            raise ValueError(f"No examples of one correctness class: {scope}/{method}")
        rows.append({
            "scope": scope, "method": method, "setting": label(a),
            "origin": a["origin"], "learning_rate": a["lr"], "steps": a["steps"],
            "num_problems": seen, "n_correct": len(c), "n_incorrect": len(w),
            "mean_correct": float(c.mean()), "mean_incorrect": float(w.mean()),
            "delta_incorrect_minus_correct": float(w.mean()-c.mean()),
            "median_correct": float(np.median(c)), "median_incorrect": float(np.median(w)),
            "std_correct": float(c.std()), "std_incorrect": float(w.std()),
            "negative_fraction_correct": float((c < 0).mean()),
            "negative_fraction_incorrect": float((w < 0).mean()),
        })
    return rows


def plot(rows, out, scope):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable

    records = [r for r in rows if r["scope"] == scope]
    lrs = sorted({r["learning_rate"] for r in records})
    steps = sorted({r["steps"] for r in records})
    by_cell = {}
    for r in records:
        key = r["learning_rate"], r["steps"]
        if key in by_cell:
            raise ValueError(f"Duplicate LR/step: {key}")
        by_cell[key] = r

    def setup(ax):
        ax.set_xlim(-.5, len(lrs)-.5)
        ax.set_ylim(-.5, len(steps)-.5)
        ax.set_xticks(range(len(lrs)), [f"{lr:.6g}" for lr in lrs],
                      rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(steps)), [str(k) for k in steps])
        ax.set_xlabel("Learning rate (sorted, categorical spacing)")
        ax.set_ylabel("SGD steps")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def missing(ax, x, y):
        ax.add_patch(Rectangle((x-.48,y-.48), .96,.96, facecolor="#eeeeee",
                               edgecolor="white", linewidth=.6))
        ax.text(x,y,"—",ha="center",va="center",color="#aaaaaa",fontsize=9)

    def tile(ax,x,y,h,value,norm,cmap):
        rgba = cmap(norm(value))
        ax.add_patch(Rectangle((x-.48,y),.96,h,facecolor=rgba,
                               edgecolor="white",linewidth=.5))
        luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
        ax.text(x,y+h/2,f"{value:.4g}",ha="center",va="center",
                fontsize=8.5,color="white" if luminance < .48 else "#17212b")

    figwidth = max(12,1.27*len(lrs)+2)
    vals = [r[k] for r in records for k in ("mean_correct","mean_incorrect")]
    if min(vals) < 0:
        extent=max(max(abs(v) for v in vals),1e-12)
        norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
        cmap=plt.get_cmap("RdBu")
    else:
        norm=Normalize(vmin=0,vmax=max(max(vals),1e-12))
        cmap=plt.get_cmap("Blues")
    fig,ax=plt.subplots(figsize=(figwidth,6.4), layout="constrained")
    for x,lr in enumerate(lrs):
        for y,k in enumerate(steps):
            r=by_cell.get((lr,k))
            if r is None:
                missing(ax,x,y)
                continue
            # y increases upward: upper half = correct, lower = incorrect.
            tile(ax,x,y,.48,r["mean_correct"],norm,cmap)
            tile(ax,x,y-.48,.48,r["mean_incorrect"],norm,cmap)
    setup(ax)
    ax.set_title(f"Mean signed penalty | {scope} | upper: correct / lower: incorrect",
                 fontsize=14,pad=14)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.83,
                 label="Original reward - updated reward")
    for ext in ("png","pdf"):
        fig.savefig(out/f"penalty_paired_{scope}.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)

    delta=[r["delta_incorrect_minus_correct"] for r in records]
    extent=max(max(abs(v) for v in delta),1e-12)
    norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
    cmap=plt.get_cmap("RdBu")
    fig,ax=plt.subplots(figsize=(figwidth,5.7),layout="constrained")
    for x,lr in enumerate(lrs):
        for y,k in enumerate(steps):
            r=by_cell.get((lr,k))
            if r is None:
                missing(ax,x,y)
                continue
            tile(ax,x,y-.48,.96,r["delta_incorrect_minus_correct"],norm,cmap)
    setup(ax)
    ax.set_title(f"Mean penalty difference (incorrect - correct) | {scope}",
                 fontsize=14,pad=14)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.83,
                 label="Incorrect minus correct (reward units)")
    for ext in ("png","pdf"):
        fig.savefig(out/f"penalty_delta_{scope}.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--top-k",type=int,default=16)
    args=p.parse_args()
    pilot=args.pilot.resolve()
    cfg=load_config(pilot)
    old=read_design(Path(cfg["old_lhs"]))
    specs=[a for a in method_specs(cfg,old) if a["method"]!="vanilla"]
    out=args.output_dir or pilot/"evaluation/penalty_analysis"
    out.mkdir(parents=True,exist_ok=True)
    rows=calculate(pilot,cfg,specs,args.top_k)
    rows.sort(key=lambda r:(r["scope"],r["learning_rate"],r["steps"]))
    write_csv(out/"penalty_by_lr_step.csv",rows)
    for scope in ("all",f"top{args.top_k}_original"):
        plot(rows,out,scope)
        best=sorted([r for r in rows if r["scope"]==scope],
                    key=lambda r:-r["delta_incorrect_minus_correct"])[:5]
        print(f"\nLargest incorrect-minus-correct gaps ({scope}):")
        for r in best:
            print(f"{r['setting']} C={r['mean_correct']:.5g} "
                  f"I={r['mean_incorrect']:.5g} "
                  f"gap={r['delta_incorrect_minus_correct']:+.5g}")
    print(f"\nSaved {len(specs)} settings: {out}",flush=True)


if __name__=="__main__":
    main()
