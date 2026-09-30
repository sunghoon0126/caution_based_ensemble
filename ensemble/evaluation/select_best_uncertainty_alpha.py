#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

DATASETS = [
    "gsm8k",
    "math500",
    "bbh",
]


QUANTILE_METHODS = [
    "median_quantile",
    "avg_quantile",
    "avg_quantile_avg",
]


ALL_METHODS = [
    "std",
    "distance",
    "median_quantile",
    "avg_quantile",
    "avg_quantile_avg",
]


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


# Fixed plotting ranges decided earlier.
Y_LIMITS = {
    "gsm8k":
        (70.0, 85.0),

    "math500":
        (15.0, 30.0),

    "bbh":
        (15.0, 30.0),
}


N_VALUES = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256, 512,
]


# ============================================================
# Alpha selection
# ============================================================

def select_best_alphas(
        summary_df,
):
    """
    Select alpha separately for each:

        dataset x quantile method

    Selection priority:

        1. highest peak_accuracy
        2. highest final_accuracy
        3. lowest degradation
        4. lowest alpha for deterministic tie-breaking
    """

    rows = []

    for dataset in DATASETS:

        for method in QUANTILE_METHODS:

            subset = summary_df[
                (
                    summary_df[
                        "dataset"
                    ] == dataset
                )
                &
                (
                    summary_df[
                        "uncertainty_method"
                    ] == method
                )
            ].copy()

            if subset.empty:
                raise RuntimeError(
                    "No alpha candidates for "
                    f"{dataset} / {method}"
                )

            # ------------------------------------------------
            # Selection rule
            # ------------------------------------------------

            subset = (
                subset
                .sort_values(
                    by=[
                        "peak_accuracy",
                        "final_accuracy",
                        "degradation",
                        "alpha",
                    ],
                    ascending=[
                        False,   # higher peak better
                        False,   # higher final better
                        True,    # lower degradation better
                        True,    # deterministic tie break
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

            best = subset.iloc[0]

            rows.append(
                {
                    "dataset":
                        dataset,

                    "uncertainty_method":
                        method,

                    "selected_alpha":
                        float(
                            best["alpha"]
                        ),

                    "config":
                        best["config"],

                    "peak_n":
                        int(
                            best["peak_n"]
                        ),

                    "peak_accuracy":
                        float(
                            best[
                                "peak_accuracy"
                            ]
                        ),

                    "final_accuracy":
                        float(
                            best[
                                "final_accuracy"
                            ]
                        ),

                    "degradation":
                        float(
                            best[
                                "degradation"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Build selected curve set
# ============================================================

def get_selected_curves(
        curves_df,
        selection_df,
        dataset,
):
    """
    Return five selected uncertainty curves:

        STD
        Distance
        selected Median-Q
        selected Avg-Q
        selected Avg-QAvg
    """

    frames = []

    # ========================================================
    # 1. STD
    # ========================================================

    std_df = curves_df[
        (
            curves_df[
                "dataset"
            ] == dataset
        )
        &
        (
            curves_df[
                "uncertainty_method"
            ] == "std"
        )
    ].copy()

    std_df[
        "plot_label"
    ] = "STD"

    std_df[
        "selected_alpha"
    ] = np.nan

    frames.append(
        std_df
    )

    # ========================================================
    # 2. Distance
    # ========================================================

    distance_df = curves_df[
        (
            curves_df[
                "dataset"
            ] == dataset
        )
        &
        (
            curves_df[
                "uncertainty_method"
            ] == "distance"
        )
    ].copy()

    distance_df[
        "plot_label"
    ] = "Distance"

    distance_df[
        "selected_alpha"
    ] = np.nan

    frames.append(
        distance_df
    )

    # ========================================================
    # 3-5. Selected alpha quantile methods
    # ========================================================

    for method in QUANTILE_METHODS:

        selection = selection_df[
            (
                selection_df[
                    "dataset"
                ] == dataset
            )
            &
            (
                selection_df[
                    "uncertainty_method"
                ] == method
            )
        ]

        if len(selection) != 1:
            raise RuntimeError(
                "Expected exactly one selected "
                f"alpha for {dataset}/{method}"
            )

        alpha = float(
            selection.iloc[0][
                "selected_alpha"
            ]
        )

        method_df = curves_df[
            (
                curves_df[
                    "dataset"
                ] == dataset
            )
            &
            (
                curves_df[
                    "uncertainty_method"
                ] == method
            )
            &
            (
                np.isclose(
                    curves_df[
                        "alpha"
                    ],
                    alpha,
                )
            )
        ].copy()

        if method_df.empty:
            raise RuntimeError(
                "Selected curve not found: "
                f"{dataset}, "
                f"{method}, "
                f"alpha={alpha}"
            )

        method_df[
            "selected_alpha"
        ] = alpha

        method_df[
            "plot_label"
        ] = (
            f"{DISPLAY_NAMES[method]} "
            f"(α={alpha:.1f})"
        )

        frames.append(
            method_df
        )

    selected_df = pd.concat(
        frames,
        ignore_index=True,
    )

    return selected_df


# ============================================================
# Plot
# ============================================================

def plot_selected_methods(
        selected_df,
        dataset,
        output_path,
):
    plt.figure(
        figsize=(9, 6)
    )

    for method in ALL_METHODS:

        subset = selected_df[
            selected_df[
                "uncertainty_method"
            ] == method
        ].copy()

        if subset.empty:
            continue

        subset = (
            subset
            .sort_values("N")
        )

        label = (
            subset[
                "plot_label"
            ]
            .iloc[0]
        )

        plt.plot(
            subset["N"],
            subset[
                "accuracy_percent_mean"
            ],
            marker="o",
            linewidth=2,
            label=label,
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

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

    plt.ylabel(
        "Accuracy (%)"
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    plt.title(
        f"{dataset.upper()} - "
        "Selected Uncertainty Metrics"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Selected method summary
# ============================================================

def build_selected_summary(
        selected_curves,
):
    rows = []

    for (
        dataset,
        method,
    ), subset in (
        selected_curves
        .groupby(
            [
                "dataset",
                "uncertainty_method",
            ]
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

        peak = (
            subset.iloc[
                peak_idx
            ]
        )

        final = (
            subset.iloc[-1]
        )

        alpha_values = (
            subset[
                "selected_alpha"
            ]
            .dropna()
            .unique()
        )

        if len(alpha_values):
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

                "selected_alpha":
                    alpha,

                "peak_n":
                    int(
                        peak["N"]
                    ),

                "peak_accuracy":
                    float(
                        peak[
                            "accuracy_percent_mean"
                        ]
                    ),

                "final_n":
                    int(
                        final["N"]
                    ),

                "final_accuracy":
                    float(
                        final[
                            "accuracy_percent_mean"
                        ]
                    ),

                "degradation":
                    float(
                        peak[
                            "accuracy_percent_mean"
                        ]
                        -
                        final[
                            "accuracy_percent_mean"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Console table
# ============================================================

def print_selection_table(
        selection_df,
):
    print()
    print(
        "=" * 95
    )

    print(
        "Selected alpha by dataset and method"
    )

    print(
        "=" * 95
    )

    print(
        f"{'Dataset':<12}"
        f"{'Method':<28}"
        f"{'Alpha':>8}"
        f"{'Peak N':>10}"
        f"{'Peak':>12}"
        f"{'Final':>12}"
        f"{'Deg.':>12}"
    )

    print(
        "-" * 95
    )

    for _, row in (
        selection_df.iterrows()
    ):

        print(
            f"{row['dataset']:<12}"
            f"{row['uncertainty_method']:<28}"
            f"{row['selected_alpha']:>8.1f}"
            f"{int(row['peak_n']):>10d}"
            f"{row['peak_accuracy']:>11.2f}%"
            f"{row['final_accuracy']:>11.2f}%"
            f"{row['degradation']:>11.2f}"
        )

    print(
        "=" * 95
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "outputs/"
            "uncertainty_metrics/"
            "aggregated_5metrics"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "uncertainty_metrics/"
            "selected_5metrics"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Paths
    # ========================================================

    curves_path = (
        args.input_dir
        / "trial_averaged_curves.csv"
    )

    summary_path = (
        args.input_dir
        / "summary_all_configs.csv"
    )

    if not curves_path.exists():
        raise FileNotFoundError(
            curves_path
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            summary_path
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = (
        args.output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load
    # ========================================================

    curves_df = pd.read_csv(
        curves_path
    )

    summary_df = pd.read_csv(
        summary_path
    )

    # ========================================================
    # Select alpha
    # ========================================================

    selection_df = (
        select_best_alphas(
            summary_df
        )
    )

    selection_path = (
        args.output_dir
        / "selected_alphas.csv"
    )

    selection_df.to_csv(
        selection_path,
        index=False,
    )

    print_selection_table(
        selection_df
    )

    # ========================================================
    # Selected curves
    # ========================================================

    selected_frames = []

    for dataset in DATASETS:

        selected_df = (
            get_selected_curves(
                curves_df=(
                    curves_df
                ),
                selection_df=(
                    selection_df
                ),
                dataset=dataset,
            )
        )

        selected_frames.append(
            selected_df
        )

        plot_path = (
            plot_dir
            / (
                f"{dataset}_"
                "selected_5metrics.png"
            )
        )

        plot_selected_methods(
            selected_df=(
                selected_df
            ),
            dataset=dataset,
            output_path=(
                plot_path
            ),
        )

    selected_curves = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    selected_curves_path = (
        args.output_dir
        / "selected_trial_averaged_curves.csv"
    )

    selected_curves.to_csv(
        selected_curves_path,
        index=False,
    )

    # ========================================================
    # Final selected summary
    # ========================================================

    selected_summary = (
        build_selected_summary(
            selected_curves
        )
    )

    selected_summary_path = (
        args.output_dir
        / "selected_summary.csv"
    )

    selected_summary.to_csv(
        selected_summary_path,
        index=False,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "Alpha selection complete"
    )
    print(
        "=" * 70
    )

    print(
        f"Selected alphas:"
    )
    print(
        f"  {selection_path}"
    )

    print(
        f"Selected curves:"
    )
    print(
        f"  {selected_curves_path}"
    )

    print(
        f"Selected summary:"
    )
    print(
        f"  {selected_summary_path}"
    )

    print(
        f"Plots:"
    )
    print(
        f"  {plot_dir}"
    )


if __name__ == "__main__":
    main()