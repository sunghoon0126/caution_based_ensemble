"""Render original and dummy-backprop curves from saved CSV."""

import argparse
import csv
from pathlib import Path


def plot(path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label in (("original_rm", "Original RM"), ("dummy_backprop_rm", "Dummy Backprop RM")):
        points = [r for r in rows if r["method"] == method]
        ax.plot([int(r["N"]) for r in points], [float(r["accuracy_percent"]) for r in points], marker="o", label=label)
    ax.set(xlabel="Selection budget N", ylabel="Accuracy (%)", title=f"Fixed dummy calibration pool: {rows[0]['dummy_calibration_pool']}")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output).mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(Path(output) / f"accuracy_vs_n.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input_path, args.output_dir)
