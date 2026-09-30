#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASETS = [
    "gsm8k",
    "math500",
    "bbh",
]

TRIALS = [
    1,
    2,
    3,
]

N_VALUES = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256, 512,
]


CONFIGS = [
    {
        "method": "std",
        "alpha": None,
        "dirname": "std",
    },
    {
        "method": "distance",
        "alpha": None,
        "dirname": "distance",
    },
]


QUANTILE_METHODS = [
    "median_quantile",
    "avg_quantile",
    "avg_quantile_avg",
]

ALPHAS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
]


for method in QUANTILE_METHODS:
    for alpha in ALPHAS:
        CONFIGS.append(
            {
                "method": method,
                "alpha": alpha,
                "dirname": (
                    f"{method}_alpha_{alpha:.1f}"
                ),
            }
        )


DISPLAY_NAMES = {
    "std":
        "STD",

    "distance":
        "Distance",

    "median_quantile":
        "Median - Quantile",

    "avg_quantile":
        "Average - Quantile",

    "avg_quantile_avg":
        "Average - Quantile Average",
}

Y_LIMITS = {
    "gsm8k": (70.0, 85.0),
    "math500": (15.0, 30.0),
    "bbh": (15.0, 30.0),
}

def load_trial_curve(
        evaluation_root,
        dataset,
        trial,
        config,
):
    path = (
        evaluation_root
        / dataset
        / f"trial{trial}"
        / config["dirname"]
        / "curves.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing curve file: {path}"
        )

    df = pd.read_csv(
        path
    )

    # We evaluated rm_neuboots only.
    df = df[
        df["method"]
        == "rm_neuboots"
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No rm_neuboots rows in {path}"
        )

    return df


def aggregate_one_config(
        evaluation_root,
        dataset,
        config,
):
    trial_frames = []

    for trial in TRIALS:

        df = load_trial_curve(
            evaluation_root=(
                evaluation_root
            ),
            dataset=dataset,
            trial=trial,
            config=config,
        )

        df = df[
            [
                "N",
                "accuracy",
                "accuracy_percent",
            ]
        ].copy()

        df["trial"] = trial

        trial_frames.append(
            df
        )

    all_trials = pd.concat(
        trial_frames,
        ignore_index=True,
    )

    grouped = (
        all_trials
        .groupby(
            "N",
            as_index=False,
        )
        .agg(
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),
            accuracy_percent_mean=(
                "accuracy_percent",
                "mean",
            ),
            accuracy_percent_std=(
                "accuracy_percent",
                "std",
            ),
        )
        .sort_values(
            "N"
        )
        .reset_index(
            drop=True
        )
    )

    grouped[
        "dataset"
    ] = dataset

    grouped[
        "uncertainty_method"
    ] = config[
        "method"
    ]

    grouped[
        "alpha"
    ] = config[
        "alpha"
    ]

    grouped[
        "config"
    ] = config[
        "dirname"
    ]

    return grouped


def build_all_aggregates(
        evaluation_root,
):
    frames = []

    for dataset in DATASETS:

        for config in CONFIGS:

            print(
                "Aggregating:",
                dataset,
                config["dirname"],
            )

            df = aggregate_one_config(
                evaluation_root=(
                    evaluation_root
                ),
                dataset=dataset,
                config=config,
            )

            frames.append(
                df
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def plot_method_dataset(
        aggregate_df,
        dataset,
        method,
        output_dir,
):
    subset = aggregate_df[
        (
            aggregate_df[
                "dataset"
            ] == dataset
        )
        &
        (
            aggregate_df[
                "uncertainty_method"
            ] == method
        )
    ].copy()

    if subset.empty:
        return

    plt.figure(
        figsize=(8, 5)
    )

    if method in {
        "std",
        "distance",
    }:

        subset = (
            subset
            .sort_values("N")
        )

        plt.plot(
            subset["N"],
            subset[
                "accuracy_percent_mean"
            ],
            marker="o",
            label=DISPLAY_NAMES[
                method
            ],
        )

    else:

        for alpha in ALPHAS:

            alpha_subset = subset[
                np.isclose(
                    subset["alpha"],
                    alpha,
                )
            ].sort_values(
                "N"
            )

            plt.plot(
                alpha_subset["N"],
                alpha_subset[
                    "accuracy_percent_mean"
                ],
                marker="o",
                label=(
                    f"alpha={alpha:.1f}"
                ),
            )

    plt.xscale(
        "log",
        base=2,
    )

    plt.xticks(
        N_VALUES,
        [
            str(n)
            for n in N_VALUES
        ],
    )

    plt.xlabel(
        "Best-of-N"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    y_min, y_max = (
        Y_LIMITS[
            dataset
        ]
    )

    plt.ylim(
        y_min,
        y_max,
    )

    plt.yticks(
        np.arange(
            y_min,
            y_max + 0.1,
            2.5,
        )
    )

    plt.title(
        f"{dataset.upper()} - "
        f"{DISPLAY_NAMES[method]}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / (
            f"{dataset}_"
            f"{method}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved plot: {output_path}"
    )


def build_summary(
        aggregate_df,
):
    rows = []

    group_columns = [
        "dataset",
        "uncertainty_method",
        "config",
    ]

    for (
        dataset,
        method,
        config,
    ), subset in (
        aggregate_df
        .groupby(
            group_columns
        )
    ):

        subset = (
            subset
            .sort_values("N")
            .reset_index(
                drop=True
            )
        )

        peak_idx = (
            subset[
                "accuracy_percent_mean"
            ]
            .to_numpy()
            .argmax()
        )

        peak_row = (
            subset.iloc[
                peak_idx
            ]
        )

        final_row = (
            subset.iloc[-1]
        )

        alpha_values = (
            subset[
                "alpha"
            ]
            .dropna()
            .unique()
        )

        if len(alpha_values) > 0:
            alpha = float(
                alpha_values[0]
            )
        else:
            alpha = np.nan

        rows.append(
            {
                "dataset":
                    dataset,

                "uncertainty_method":
                    method,

                "alpha":
                    alpha,

                "config":
                    config,

                "peak_n":
                    int(
                        peak_row["N"]
                    ),

                "peak_accuracy":
                    float(
                        peak_row[
                            "accuracy_percent_mean"
                        ]
                    ),

                "final_n":
                    int(
                        final_row["N"]
                    ),

                "final_accuracy":
                    float(
                        final_row[
                            "accuracy_percent_mean"
                        ]
                    ),

                "degradation":
                    float(
                        peak_row[
                            "accuracy_percent_mean"
                        ]
                        -
                        final_row[
                            "accuracy_percent_mean"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path(
            "outputs/"
            "uncertainty_metrics/"
            "evaluation_5metrics"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "uncertainty_metrics/"
            "aggregated_5metrics"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = (
        args.output_dir
        / "plots_all_alpha"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Trial average
    # ========================================================

    aggregate_df = (
        build_all_aggregates(
            args.evaluation_root
        )
    )

    aggregate_path = (
        args.output_dir
        / "trial_averaged_curves.csv"
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    # ========================================================
    # Peak / final summary
    # ========================================================

    summary_df = (
        build_summary(
            aggregate_df
        )
    )

    summary_path = (
        args.output_dir
        / "summary_all_configs.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Plots
    # ========================================================

    for dataset in DATASETS:

        for method in DISPLAY_NAMES:

            plot_method_dataset(
                aggregate_df=(
                    aggregate_df
                ),
                dataset=dataset,
                method=method,
                output_dir=(
                    plot_dir
                ),
            )

    print()
    print("=" * 70)
    print("Aggregation complete")
    print("=" * 70)

    print(
        "Trial averaged curves:"
    )
    print(
        f"  {aggregate_path}"
    )

    print(
        "Summary:"
    )
    print(
        f"  {summary_path}"
    )

    print(
        "Plots:"
    )
    print(
        f"  {plot_dir}"
    )


if __name__ == "__main__":
    main()