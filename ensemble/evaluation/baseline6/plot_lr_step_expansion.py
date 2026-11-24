"""Readable Baseline 6 LR x Step plots from existing evaluation CSVs.

No model loading, GPUs, rescoring, or re-evaluation. All scores are updated RM
rewards (no lambda). Selection of featured arms uses TUNE N=max only; the
HOLDOUT set is never used to select the featured arms.

Charts for all/tune/holdout:
  focused_accuracy_vs_n_<split>.png   Vanilla (bold) vs 3 best tuned arms
  overview_accuracy_vs_n_<split>.png  All arms gray, vanilla bold, top-3 colored
  n256_delta_ranked_<split>.png       All arms' N=256 delta vs vanilla (pp)
  lr_step_delta_heatmap_<split>.png   New grid difference vs vanilla N=256
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("all", "tune", "holdout")
ACCENT = ("#2563eb", "#e07b12", "#16a085")
BASELINE = "#111827"
OTHER = "#94a3b8"


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prettify(method_row):
    origin = method_row["origin"]
    if origin == "vanilla":
        return "Vanilla RM (no update)"
    lr = float(method_row["learning_rate"])
    step = int(method_row["steps"])
    source = {"test1": "Test 1", "old_lhs": "Old LHS", "new": "New grid"}.get(
        origin, origin
    )
    return f"LR={lr:.2g}, steps={step} [{source}]"


def save(fig, out, name, dpi):
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_curves(dataset, meta, chosen, subset, out, dpi, overview):
    by_method = dataset[subset]
    ns = sorted(next(iter(by_method.values())))
    vanilla = by_method["vanilla"]
    methods = [method for method in by_method if method != "vanilla"]
    nmax = ns[-1]

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    if overview:
        for method in methods:
            data = by_method[method]
            ax.plot(ns, [data[n] for n in ns], color=OTHER, alpha=0.18,
                    linewidth=0.9, zorder=1)
    ax.plot(ns, [vanilla[n] for n in ns], color="white", linewidth=5.6,
            zorder=7, solid_capstyle="round")
    ax.plot(ns, [vanilla[n] for n in ns], color=BASELINE, linewidth=3.3,
            linestyle="--", marker="s", markersize=6.0, zorder=8,
            label="Vanilla RM — baseline")

    if "test1_s1" in by_method and "test1_s1" not in chosen:
        previous = by_method["test1_s1"]
        ax.plot(ns, [previous[n] for n in ns], color="#7c3aed",
                linewidth=1.8, linestyle=":", marker="D", markersize=3.8,
                alpha=0.85, zorder=4, label="Previous SGD (LR=1e-6, 1 step)")

    for i, method in enumerate(chosen):
        data = by_method[method]
        ax.plot(ns, [data[n] for n in ns], color=ACCENT[i], linewidth=2.5,
                marker="o", markersize=4.9, zorder=5-i,
                label=prettify(meta[method]))

    base_final = vanilla[nmax]
    ax.annotate(f"Vanilla at N={nmax}: {base_final:.1f}%",
                xy=(nmax, base_final), xytext=(-9, -17),
                textcoords="offset points", ha="right", va="top",
                color=BASELINE, fontsize=9, fontweight="bold",
                bbox={"boxstyle": "round,pad=.25", "fc": "white",
                      "ec": "#cbd5e1", "alpha": 0.92}, zorder=10)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Best-of-N budget")
    ax.set_ylabel("Accuracy (%)")
    suffix = "All conditions (faint) + top 3" if overview else "Baseline + top 3"
    ax.set_title(f"{subset.capitalize()} | Updated-reward BoN | {suffix}")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=9, framealpha=0.96)
    ax.margins(y=0.12)
    fig.tight_layout()
    prefix = "overview" if overview else "focused"
    save(fig, out, f"{prefix}_accuracy_vs_n_{subset}", dpi)


def plot_nmax_delta(rows, meta, chosen, subset, out, dpi):
    subset_rows = [r for r in rows if r["subset"] == subset]
    by_method = {r["method"]: r for r in subset_rows}
    base = float(by_method["vanilla"]["final_accuracy_percent"])
    others = [r for r in subset_rows if r["origin"] != "vanilla"]
    others.sort(key=lambda r: float(r["final_accuracy_percent"]) - base)
    values = [float(r["final_accuracy_percent"]) - base for r in others]
    labels = [f"{r['method']}  ({float(r['learning_rate']):.2g}, "
              f"k={int(r['steps'])})" for r in others]
    colors = [ACCENT[chosen.index(r["method"])]
              if r["method"] in chosen else
              ("#8596af" if r["origin"] == "new" else "#cbd5e1")
              for r in others]
    fig, ax = plt.subplots(figsize=(11, max(7.2, .30*len(others)+2)))
    positions = np.arange(len(others))
    ax.barh(positions, values, color=colors, height=0.77, zorder=3)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.axvline(0, color=BASELINE, linewidth=2.4, linestyle="--",
               zorder=5, label=f"Vanilla RM = {base:.1f}% (0 pp)")
    ax.set_xlabel("Accuracy change vs Vanilla at N=max (percentage points)")
    ax.set_title(f"{subset.capitalize()} | All LR/Step settings vs Vanilla RM")
    ax.grid(axis="x", alpha=0.18, zorder=0)
    ax.legend(fontsize=9)
    xmax = max(1, max(abs(v) for v in values))
    ax.set_xlim(min(values + [0]) - xmax*.22, max(values + [0]) + xmax*.22)
    for k, value in enumerate(values):
        ax.text(value + .02*xmax if value >= 0 else value - .02*xmax, k,
                f"{value:+.1f}", va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7.5, color=BASELINE)
    fig.tight_layout()
    save(fig, out, f"n256_delta_ranked_{subset}", dpi)


def plot_new_grid_heatmap(rows, subset, out, dpi):
    group = [r for r in rows if r["subset"] == subset]
    vanilla = next(r for r in group if r["origin"] == "vanilla")
    base = float(vanilla["final_accuracy_percent"])
    grid = [r for r in group if r["origin"] == "new"]
    if not grid:
        return
    lrs = sorted({float(r["learning_rate"]) for r in grid})
    steps = sorted({int(r["steps"]) for r in grid})
    matrix = np.full((len(lrs), len(steps)), np.nan)
    for r in grid:
        i = lrs.index(float(r["learning_rate"]))
        j = steps.index(int(r["steps"]))
        if np.isfinite(matrix[i, j]):
            raise ValueError("Duplicate new-grid LR/step combination")
        matrix[i, j] = float(r["final_accuracy_percent"]) - base
    if not np.isfinite(matrix).all():
        raise ValueError("Missing new-grid LR/step score")
    magnitude = max(float(np.max(np.abs(matrix))), 0.1)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, .65*len(lrs)+2.4)))
    img = ax.imshow(matrix, cmap="RdBu", vmin=-magnitude, vmax=magnitude,
                    aspect="auto")
    ax.set_xticks(range(len(steps)), [str(v) for v in steps])
    ax.set_yticks(range(len(lrs)), [f"{v:.2g}" for v in lrs])
    ax.set_xlabel("SGD steps")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"{subset.capitalize()} | N=max delta vs Vanilla ({base:.1f}%)")
    for i in range(len(lrs)):
        for j in range(len(steps)):
            value = matrix[i, j]
            color = "white" if abs(value) > .53*magnitude else BASELINE
            ax.text(j, i, f"{value:+.1f}pp", ha="center",
                    va="center", fontsize=10, color=color, fontweight="bold")
    fig.colorbar(img, ax=ax, label="Accuracy − Vanilla (percentage points)")
    fig.tight_layout()
    save(fig, out, f"lr_step_delta_heatmap_{subset}", dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", required=True, type=Path,
                   help="Directory containing summary.csv and curves.csv")
    p.add_argument("--output-dir", type=Path,
                   help="Defaults to <evaluation-dir>/readable_plots")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--splits", nargs="+", choices=SPLITS,
                   default=["holdout"],
                   help="Default: only holdout. For all use --splits all tune holdout")
    args = p.parse_args()
    if not 1 <= args.top_k <= len(ACCENT):
        p.error(f"--top-k must be 1..{len(ACCENT)}")
    if args.dpi < 50:
        p.error("--dpi must be >=50")

    summaries = load_csv(args.evaluation_dir / "summary.csv")
    curves = load_csv(args.evaluation_dir / "curves.csv")
    meta = {r["method"]: r for r in summaries if r["subset"] == "all"}
    by_split = defaultdict(lambda: defaultdict(dict))
    for r in curves:
        subset, method, n = r["subset"], r["method"], int(r["N"])
        if n in by_split[subset][method]:
            raise ValueError(f"Duplicate curve point {subset}/{method}/{n}")
        by_split[subset][method][n] = float(r["accuracy_percent"])
    for subset in SPLITS:
        if "vanilla" not in by_split[subset]:
            raise ValueError(f"Vanilla baseline missing in {subset}")
        expected_ns = set(by_split[subset]["vanilla"])
        for method, points in by_split[subset].items():
            if set(points) != expected_ns:
                raise ValueError(f"Inconsistent N points: {subset}/{method}")

    tune = [r for r in summaries if r["subset"] == "tune"
            and r["method"] != "vanilla"]
    tune.sort(key=lambda r: (-float(r["final_accuracy_percent"]),
                              int(r["steps"]), float(r["learning_rate"]),
                              r["method"]))
    chosen = [r["method"] for r in tune[:args.top_k]]
    if len(chosen) < args.top_k:
        raise ValueError("Not enough tuned settings")
    for method in chosen:
        if any(method not in by_split[subset] for subset in SPLITS):
            raise ValueError(f"Featured method {method} missing in a split")
    out = args.output_dir or args.evaluation_dir / "readable_plots"
    out.mkdir(parents=True, exist_ok=True)

    for subset in args.splits:
        plot_curves(by_split, meta, chosen, subset, out, args.dpi, overview=False)
        plot_curves(by_split, meta, chosen, subset, out, args.dpi, overview=True)
        plot_nmax_delta(summaries, meta, chosen, subset, out, args.dpi)
        plot_new_grid_heatmap(summaries, subset, out, args.dpi)

    nmax = max(by_split["all"]["vanilla"])
    print("Featured (selected using TUNE N=max only; no holdout selection):")
    for method in chosen:
        setting = meta[method]
        acc = by_split["holdout"][method][nmax]
        baseline = by_split["holdout"]["vanilla"][nmax]
        print(f"  {prettify(setting)} | "
              f"holdout N={nmax} {acc:.2f}% vs vanilla {baseline:.2f}% "
              f"({acc-baseline:+.2f} pp)")
    print(f"Plots saved: {out}")


if __name__ == "__main__":
    main()
