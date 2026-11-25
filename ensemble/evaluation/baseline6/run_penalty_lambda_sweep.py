"""Lambda sweep for an explicitly chosen cached LR/step pilot arm.

Reuses the existing lambda_pilot implementation and the original tune/holdout
split. Writes to an independent output directory; never touches GPU worker
manifests or the existing Step 2 sweep. No model loading or adaptation.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from ensemble.evaluation.baseline6.run_selected_sgd import (
    load_manifest, lambda_pilot, read_csv, save_figure, label,
)


def equal(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15)


def plot_final(out, setting, best_lambda):
    curves = read_csv(out / "lambda_curves.csv")
    choices = [(0.0, "Vanilla RM (lambda=0)"),
               (0.5, "Updated RM (lambda=0.5)")]
    if not any(equal(best_lambda, lam) for lam, _ in choices):
        choices.append((best_lambda, f"Best lambda ({best_lambda:g})"))
    else:
        choices = [(lam, name + (" [best]" if equal(lam, best_lambda) else ""))
                   for lam, name in choices]

    colors = ("#59677b", "#2563eb", "#e07b12")
    styles = ("-", "--", "-.")
    markers = ("o", "s", "^")
    offsets = (-6, 0, 6)
    for split in ("all", "tune", "holdout"):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        handles = []
        for i, (lam, name) in enumerate(choices):
            rows = sorted([r for r in curves if r["subset"] == split
                           and equal(r["lambda"], lam)], key=lambda r: int(r["N"]))
            if not rows:
                raise ValueError(f"No curve for {split}, lambda={lam}")
            ns = [int(r["N"]) for r in rows]
            ys = [float(r["accuracy_percent"]) for r in rows]
            if lam == 0:
                ax.plot(ns, ys, color=colors[i], linewidth=6, alpha=.12, zorder=1)
            ax.plot(ns, ys, color=colors[i], linestyle=styles[i], linewidth=1.8,
                    alpha=.95, zorder=2+i)
            offset = ScaledTranslation(offsets[i]/72, 0, fig.dpi_scale_trans)
            ax.plot(ns, ys, linestyle="none", marker=markers[i], color=colors[i],
                    markeredgecolor="white", markeredgewidth=.8, markersize=7,
                    transform=ax.transData+offset, zorder=10+i)
            handles.append(Line2D([0], [0], color=colors[i], linestyle=styles[i],
                                  linewidth=1.8, marker=markers[i], markersize=7,
                                  label=f"{name} | N={ns[-1]}: {ys[-1]:.2f}%"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlim(ns[0]/1.25, ns[-1]*1.25)
        ax.set(xlabel="Best-of-N", ylabel="Accuracy (%)",
               title=f"Lambda comparison | {split} | {label(setting)}")
        ax.grid(alpha=.2)
        ax.legend(handles=handles)
        fig.tight_layout()
        save_figure(fig, out, f"lambda_final_comparison_{split}")


def compare_sweeps(source, out, chosen):
    old_path = source / "lambda_n256.csv"
    old_json = source / "selected_lambda.json"
    if not old_path.is_file() or not old_json.is_file():
        print("Previous lambda sweep not found; skipping comparison.", flush=True)
        return
    old_meta = json.loads(old_json.read_text())
    previous = old_meta["setting"]
    old_best = float(old_meta["selected_lambda"])
    new_best = float(json.loads((out/"selected_lambda.json").read_text())["selected_lambda"])
    if old_meta.get("formula") != "(1-lambda)*r-lambda*(r-r_updated)":
        raise ValueError("Previous lambda score formula differs")
    old, new = read_csv(old_path), read_csv(out/"lambda_n256.csv")
    for split in ("all", "tune", "holdout"):
        data = []
        for rows in (old, new):
            pts = sorted([r for r in rows if r["subset"] == split],
                         key=lambda r: float(r["lambda"]))
            data.append(([float(r["lambda"]) for r in pts],
                         [float(r["accuracy_percent"]) for r in pts],
                         [int(r["N"]) for r in pts],
                         [int(r["num_problems"]) for r in pts]))
        x0, y0, n0, c0 = data[0]
        x1, y1, n1, c1 = data[1]
        if not x0 or x0 != x1 or n0 != n1 or c0 != c1:
            raise ValueError(f"Sweep grid or sample size mismatch: {split}")
        if not equal(y0[x0.index(0.0)], y1[x1.index(0.0)]):
            raise ValueError(f"Vanilla accuracies differ: {split}")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, (x, y, _, _) in enumerate(data):
            setting = previous if i == 0 else chosen
            best = old_best if i == 0 else new_best
            color = ("#2563eb", "#e07b12")[i]
            style = ("-", "--")[i]
            marker = ("o", "s")[i]
            ax.plot(x, y, color=color, linestyle=style, linewidth=1.8,
                    label=f"{label(setting)} (tune-selected lambda={best:g})")
            off = ScaledTranslation((-3 if i == 0 else 3)/72, 0, fig.dpi_scale_trans)
            ax.plot(x, y, linestyle="none", marker=marker, markersize=5.5,
                    color=color, markeredgecolor="white", markeredgewidth=.65,
                    transform=ax.transData+off, zorder=5+i)
        vanilla = y0[x0.index(0.0)]
        ax.axhline(vanilla, color="#59677b", linewidth=1.2, linestyle=":",
                   label=f"Vanilla RM ({vanilla:.2f}%)")
        ax.set(xlabel="Lambda", ylabel=f"N={n0[0]} accuracy (%)",
               title=f"Step comparison | {split}")
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out, f"lambda_step_comparison_n256_{split}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, default=7e-7)
    p.add_argument("--steps", type=int, default=16)
    args = p.parse_args()
    source = args.selection_dir.resolve()
    out = args.output_dir.resolve()
    if out == source:
        raise ValueError("Use a separate output directory for the new lambda sweep")
    manifest = load_manifest(source)
    if manifest.get("transfer_dataset"):
        raise ValueError("Expected the original GSM8K pilot selection manifest")
    matches = [a for a in manifest["selected"]
               if equal(a["lr"], args.learning_rate) and a["steps"] == args.steps]
    if len(matches) != 1:
        raise ValueError("Requested setting must occur exactly once in selected_config.json")
    chosen = matches[0]
    out.mkdir(parents=True, exist_ok=True)
    config = {"selection_dir": str(source), "setting": chosen,
              "pilot": manifest["pilot"], "pilot_config": manifest["pilot_config"]}
    conf = out/"lambda_sweep_config.json"
    if conf.exists() and json.loads(conf.read_text()) != config:
        raise ValueError("Output directory already has a different sweep config")
    conf.write_text(json.dumps(config, indent=2))
    print(f"CPU lambda sweep for {label(chosen)}", flush=True)
    # lambda_pilot uses selected[0]; explicitly choose Step 16 without editing
    # the original selected_config.json used by ongoing GPU workers.
    lambda_pilot(dict(manifest, selected=[chosen]), out)
    result = json.loads((out/"selected_lambda.json").read_text())
    plot_final(out, chosen, float(result["selected_lambda"]))
    compare_sweeps(source, out, chosen)
    print(f"Saved results and plots to {out}", flush=True)


if __name__ == "__main__":
    main()
