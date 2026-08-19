#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = [
    "rm",
    "pessimism",
    "caution",
    "neuboots_pessimism",
    "rm_neuboots",
]


METHOD_NAMES = {
    "rm": "Reward Model",
    "pessimism": "RND Pessimism",
    "caution": "RM + RND",
    "neuboots_pessimism": "NeuBoots Pessimism",
    "rm_neuboots": "RM + NeuBoots",
}


N_VALUES = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256, 512,
]


def load_curve(path):
    df = pd.read_csv(path)

    required = {
        "dataset",
        "method",
        "N",
        "accuracy_percent",
        "ci_low",
        "ci_high",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    return df


def plot_dataset(
    ax,
    df,
    dataset_name,
    methods,
):
    subset = df[
        df["dataset"] == dataset_name
    ]

    for method in methods:

        curve = subset[
            subset["method"] == method
        ].sort_values("N")

        if curve.empty:
            continue

        x = curve["N"].to_numpy()
        y = curve[
            "accuracy_percent"
        ].to_numpy()

        low = curve[
            "ci_low"
        ].to_numpy()

        high = curve[
            "ci_high"
        ].to_numpy()

        label = METHOD_NAMES.get(
            method,
            method,
        )

        line, = ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
        )

        ax.fill_between(
            x,
            low,
            high,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xticks(
        N_VALUES
    )

    ax.set_xticklabels(
        [str(n) for n in N_VALUES]
    )

    ax.set_xlabel(
        "Number of candidates N"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        dataset_name
    )

    ax.grid(
        True,
        alpha=0.25,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gsm8k",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--math500",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--bbh",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=[
            "rm",
            "pessimism",
            "caution",
        ],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/fig3_eval/figure3.png"
        ),
    )

    args = parser.parse_args()

    datasets = []

    if args.gsm8k is not None:
        datasets.append(
            (
                "GSM8K",
                load_curve(args.gsm8k),
            )
        )

    if args.math500 is not None:
        datasets.append(
            (
                "MATH-500",
                load_curve(args.math500),
            )
        )

    if args.bbh is not None:
        datasets.append(
            (
                "BBH",
                load_curve(args.bbh),
            )
        )

    if not datasets:
        raise ValueError(
            "At least one dataset curve "
            "must be provided."
        )

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(
            5 * len(datasets),
            4.5,
        ),
        squeeze=False,
    )

    axes = axes[0]

    for ax, (
        dataset_name,
        df,
    ) in zip(
        axes,
        datasets,
    ):
        plot_dataset(
            ax=ax,
            df=df,
            dataset_name=dataset_name,
            methods=args.methods,
        )

    axes[0].legend(
        loc="best"
    )

    fig.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
    )

    pdf_path = (
        args.output
        .with_suffix(".pdf")
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved PNG: {args.output}"
    )

    print(
        f"Saved PDF: {pdf_path}"
    )


if __name__ == "__main__":
    main()