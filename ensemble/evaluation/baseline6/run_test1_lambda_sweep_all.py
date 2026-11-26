"""CPU-only lambda sweep over all 500x256 Test 1 cached candidates."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ensemble.evaluation.baseline6.common import groups, finite
from ensemble.evaluation.baseline6.run_selected_sgd import (
    LAMBDAS, add_problem, summarize, write_csv, save_figure,
)


def plots(out, curves, finals, best, lr, steps, ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation

    title = f"GSM8K | LR={lr:g}, Steps={steps} | All 500 problems"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = [r["lambda"] for r in finals]
    ys = [r["accuracy_percent"] for r in finals]
    ax.plot(xs, ys, color="#2563eb", marker="o", markersize=4, linewidth=1.8)
    ax.axhline(ys[xs.index(0.0)], color="#64748b", linestyle=":",
               label="Vanilla RM")
    ax.axvline(.5, color="#64748b", linestyle="--", alpha=.7,
               label="Updated RM (lambda=0.5)")
    ax.scatter([best["lambda"]], [best["accuracy_percent"]], s=85,
               color="#e07b12", zorder=5,
               label=f"Best lambda={best['lambda']:g}: {best['accuracy_percent']:.2f}%")
    ax.set(xlabel="Lambda", ylabel=f"N={ns[-1]} accuracy (%)", title=title)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out, "lambda_accuracy_n256")

    choices = [(0., "Vanilla RM (lambda=0)"), (.5, "Updated RM (lambda=0.5)")]
    if best["lambda"] not in (0., .5):
        choices.append((best["lambda"], f"Best lambda ({best['lambda']:g})"))
    else:
        choices = [(lam, name+(" [best]" if lam == best["lambda"] else ""))
                   for lam, name in choices]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    handles = []
    for i, (lam, name) in enumerate(choices):
        pts = sorted([r for r in curves if r["lambda"] == lam], key=lambda r:r["N"])
        x, y = [r["N"] for r in pts], [r["accuracy_percent"] for r in pts]
        color = ("#59677b", "#2563eb", "#e07b12")[i]
        style = ("-", "--", "-.")[i]
        marker = ("o", "s", "^")[i]
        if i == 0:
            ax.plot(x, y, color=color, linewidth=6, alpha=.12, zorder=1)
        ax.plot(x, y, color=color, linestyle=style, linewidth=1.8, zorder=2+i)
        off = ScaledTranslation((-6, 0, 6)[i]/72, 0, fig.dpi_scale_trans)
        ax.plot(x, y, color=color, linestyle="none", marker=marker,
                markeredgecolor="white", markeredgewidth=.8, markersize=7,
                transform=ax.transData+off, zorder=10+i)
        handles.append(Line2D([0],[0],color=color,linestyle=style,marker=marker,
                              label=f"{name} | N={ns[-1]}: {y[-1]:.2f}%"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlim(ns[0]/1.25, ns[-1]*1.25)
    ax.set(xlabel="Best-of-N", ylabel="Accuracy (%)", title=title)
    ax.grid(alpha=.2)
    ax.legend(handles=handles)
    fig.tight_layout()
    save_figure(fig, out, "lambda_final_comparison_all")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test1", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root, out = args.test1.resolve(), args.output_dir.resolve()
    if root == out:
        raise ValueError("Use a separate output directory")
    cfg = json.loads((root/"test1_config.json").read_text())
    num, nmax, dummy_pool = (int(cfg[k]) for k in
                             ("max_problems", "scored_candidates", "dummy_pool"))
    lr, steps = float(cfg["learning_rate"]), 8
    if (num, nmax, dummy_pool) != (500, 256, 512):
        raise ValueError("Expected Test 1 500x256, dummy pool 512")
    if not math.isclose(lr, 1e-6, rel_tol=1e-12) or steps not in cfg["steps"]:
        raise ValueError("Expected LR 1e-6 and Step 8 checkpoint")
    ns = [2**i for i in range(nmax.bit_length())]
    names = [f"{lam:g}" for lam in LAMBDAS]
    counts, totals = defaultdict(Counter), Counter()
    setting = {"method": "step8_lr1e-06", "lr": lr, "steps": steps, "origin": "test1"}
    config = {"setting": setting, "test1": str(root), "test1_config": cfg,
              "selection_subset": "all", "lambdas": LAMBDAS,
              "formula": "(1-lambda)*r-lambda*(r-r_updated)"}
    out.mkdir(parents=True, exist_ok=True)
    conf = out/"lambda_sweep_config.json"
    if conf.exists() and json.loads(conf.read_text()) != config:
        raise ValueError("Output directory contains a different config")
    conf.write_text(json.dumps(config, indent=2))
    row_config = None
    for i, batch in enumerate(groups(root/"step_scores.jsonl", nmax)):
        if i >= num or int(batch[0]["dataset_index"]) != i:
            raise ValueError("Unexpected dataset index or extra problem")
        for row in batch:
            bc = row["backprop_config"]
            if row_config is None:
                row_config = bc
            if (bc != row_config or bc["optimizer"] != "sgd"
                    or not math.isclose(float(bc["learning_rate"]), lr, rel_tol=1e-12)
                    or steps not in bc["steps"] or not row.get("finite")
                    or bc["dummy_mode"] != "fixed_min"
                    or bc["dummy_calibration_pool"] != dummy_pool
                    or bc["scored_candidates_per_problem"] != nmax
                    or bc["max_problems"] != num
                    or row["correct"] not in (0, 1)
                    or row["dataset"] != "gsm8k"):
                raise ValueError(f"Unexpected cached settings at {row['candidate_id']}")
        orig = np.array([finite(r["reward_model_score"]) for r in batch])
        updated = np.array([finite(r["scores_by_step"][str(steps)]) for r in batch])
        penalty = orig-updated
        truth = [int(r["correct"]) for r in batch]
        scores = {name: (.5*updated if lam == .5 else (1-lam)*orig-lam*penalty)
                  for name, lam in zip(names, LAMBDAS)}
        add_problem(counts, totals, ("all",), truth, scores, ns)
        if (i+1) % 25 == 0 or i == 0:
            print(f"Loaded {i+1}/{num} problems", flush=True)
    if totals["all"] != num:
        raise ValueError(f"Incomplete Test 1 cache: {totals['all']}/{num}")
    curves = summarize(counts, totals, names, ns)
    for r in curves:
        r["lambda"] = float(r.pop("method"))
    finals = sorted([r for r in curves if r["N"] == nmax], key=lambda r:r["lambda"])
    best = min(finals, key=lambda r:(-r["accuracy_percent"], r["lambda"]))
    write_csv(out/"lambda_curves.csv", curves)
    write_csv(out/"lambda_n256.csv", finals)
    (out/"selected_lambda.json").write_text(json.dumps({
        "setting": setting, "selected_lambda": best["lambda"], "all": best,
        "formula": config["formula"],
        "selection_metric": "All N=256 accuracy (in-sample)",
    }, indent=2))
    summaries = []
    for lam in LAMBDAS:
        pts = [r for r in curves if r["lambda"] == lam]
        peak, last = max(pts,key=lambda r:r["accuracy_percent"]), pts[-1]
        summaries.append(dict(last, peak_n=peak["N"], peak_accuracy_percent=peak["accuracy_percent"],
                              degradation_pp=peak["accuracy_percent"]-last["accuracy_percent"]))
    write_csv(out/"lambda_summary.csv", summaries)
    plots(out, curves, finals, best, lr, steps, ns)
    for r in sorted(finals,key=lambda r:(-r["accuracy_percent"],r["lambda"])):
        print(f"lambda={r['lambda']:g} N=256 acc={r['accuracy_percent']:.2f}% "
              f"delta={r['delta_vs_vanilla_pp']:+.2f}pp")
    print(f"Best lambda={best['lambda']:g}; all accuracy={best['accuracy_percent']:.2f}%")
    print(f"Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
