#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)


# ============================================================
# Configuration
# ============================================================

METHODS = {
    "Caution": "caution_uncertainty",
    "NeuBoots": "neuboots_uncertainty",
}


LENGTH_COLUMNS = [
    "response_char_length",
    "response_word_length",
    "response_token_length",
]


# ============================================================
# Correlation
# ============================================================

def compute_correlation(
        df,
        x_col,
        y_col,
):

    subset = (
        df[
            [
                x_col,
                y_col,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    x = subset[
        x_col
    ].to_numpy()

    y = subset[
        y_col
    ].to_numpy()

    if len(x) < 2:
        return {
            "n": len(x),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson = pearsonr(
        x,
        y,
    )

    spearman = spearmanr(
        x,
        y,
    )

    return {
        "n":
            len(x),

        "pearson_r":
            float(
                pearson.statistic
            ),

        "pearson_p":
            float(
                pearson.pvalue
            ),

        "spearman_rho":
            float(
                spearman.statistic
            ),

        "spearman_p":
            float(
                spearman.pvalue
            ),
    }


def build_correlation_table(df):

    rows = []

    trials = sorted(
        df[
            "trial"
        ].unique()
    )

    groups = [
        ("pooled", df)
    ]

    for trial in trials:

        groups.append(
            (
                f"trial{trial}",
                df[
                    df["trial"]
                    == trial
                ],
            )
        )

    # ========================================================
    # Main correlation table
    # ========================================================

    for (
        group_name,
        group_df,
    ) in groups:

        for length_col in (
            LENGTH_COLUMNS
        ):

            for (
                method_name,
                uncertainty_col,
            ) in METHODS.items():

                result = (
                    compute_correlation(
                        df=group_df,
                        x_col=length_col,
                        y_col=(
                            uncertainty_col
                        ),
                    )
                )

                rows.append(
                    {
                        "group":
                            group_name,

                        "length_measure":
                            length_col,

                        "method":
                            method_name,

                        **result,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Truncation analysis
# ============================================================

def build_truncation_table(
        df,
        length_col,
):

    rows = []

    groups = {
        "all":
            df,

        "not_truncated":
            df[
                df[
                    "truncated_512"
                ] == 0
            ],

        "truncated":
            df[
                df[
                    "truncated_512"
                ] == 1
            ],
    }

    for (
        subset_name,
        subset,
    ) in groups.items():

        for (
            method_name,
            uncertainty_col,
        ) in METHODS.items():

            result = (
                compute_correlation(
                    df=subset,
                    x_col=length_col,
                    y_col=uncertainty_col,
                )
            )

            rows.append(
                {
                    "subset":
                        subset_name,

                    "method":
                        method_name,

                    "length_measure":
                        length_col,

                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Length-bin statistics
# ============================================================

def build_bin_table(
        df,
        length_col,
        bin_size,
):

    work = df.copy()

    work["length_bin_start"] = (
        np.floor(
            work[
                length_col
            ]
            / bin_size
        )
        * bin_size
    ).astype(int)

    work["length_bin_end"] = (
        work[
            "length_bin_start"
        ]
        + bin_size
        - 1
    )

    rows = []

    for (
        method_name,
        uncertainty_col,
    ) in METHODS.items():

        grouped = (
            work
            .groupby(
                [
                    "length_bin_start",
                    "length_bin_end",
                ],
                as_index=False,
            )
            .agg(
                count=(
                    uncertainty_col,
                    "size",
                ),

                mean_response_length=(
                    length_col,
                    "mean",
                ),

                mean_uncertainty=(
                    uncertainty_col,
                    "mean",
                ),

                median_uncertainty=(
                    uncertainty_col,
                    "median",
                ),

                std_uncertainty=(
                    uncertainty_col,
                    "std",
                ),
            )
        )

        grouped[
            "method"
        ] = method_name

        grouped[
            "sem_uncertainty"
        ] = (
            grouped[
                "std_uncertainty"
            ]
            / np.sqrt(
                grouped[
                    "count"
                ]
            )
        )

        rows.append(
            grouped
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Hexbin graph
# ============================================================

def plot_hexbin(
        df,
        length_col,
        uncertainty_col,
        method_name,
        output_path,
):

    x = df[
        length_col
    ].to_numpy()

    y = df[
        uncertainty_col
    ].to_numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=70,
        mincnt=1,
        bins="log",
    )

    fig.colorbar(
        hb,
        ax=ax,
        label="log10(candidate count)",
    )

    ax.set_xlabel(
        "Response length (tokens)"
    )

    ax.set_ylabel(
        f"{method_name} uncertainty"
    )

    ax.set_title(
        f"Response Length vs "
        f"{method_name} Uncertainty"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


# ============================================================
# Binned graph
# ============================================================

def plot_binned_curve(
        bin_df,
        method_name,
        min_bin_count,
        output_path,
):

    subset = (
        bin_df[
            bin_df[
                "method"
            ] == method_name
        ]
        .copy()
    )

    subset = subset[
        subset[
            "count"
        ] >= min_bin_count
    ]

    subset = subset.sort_values(
        "mean_response_length"
    )

    x = subset[
        "mean_response_length"
    ].to_numpy()

    y = subset[
        "mean_uncertainty"
    ].to_numpy()

    sem = subset[
        "sem_uncertainty"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        x,
        y,
        marker="o",
        markersize=4,
    )

    ax.fill_between(
        x,
        y - sem,
        y + sem,
        alpha=0.2,
    )

    ax.set_xlabel(
        "Response length (tokens)"
    )

    ax.set_ylabel(
        f"Mean {method_name} uncertainty"
    )

    ax.set_title(
        f"Mean {method_name} Uncertainty "
        f"by Response Length"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


# ============================================================
# Standardized comparison curve
# ============================================================

def plot_standardized_comparison(
        bin_df,
        min_bin_count,
        output_path,
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for method_name in METHODS:

        subset = (
            bin_df[
                bin_df[
                    "method"
                ] == method_name
            ]
            .copy()
        )

        subset = subset[
            subset[
                "count"
            ] >= min_bin_count
        ]

        subset = subset.sort_values(
            "mean_response_length"
        )

        values = subset[
            "mean_uncertainty"
        ].to_numpy()

        mean_value = (
            values.mean()
        )

        std_value = (
            values.std(
                ddof=0
            )
        )

        if std_value == 0:
            standardized = (
                np.zeros_like(
                    values
                )
            )
        else:
            standardized = (
                (
                    values
                    - mean_value
                )
                / std_value
            )

        ax.plot(
            subset[
                "mean_response_length"
            ],
            standardized,
            marker="o",
            markersize=4,
            label=method_name,
        )

    ax.set_xlabel(
        "Response length (tokens)"
    )

    ax.set_ylabel(
        "Standardized mean uncertainty"
    )

    ax.set_title(
        "Response Length vs "
        "Standardized Uncertainty"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


# ============================================================
# Console summary
# ============================================================

def print_main_table(
        correlation_df,
):

    subset = correlation_df[
        (
            correlation_df[
                "group"
            ] == "pooled"
        )
        &
        (
            correlation_df[
                "length_measure"
            ]
            == "response_token_length"
        )
    ]

    print()
    print("=" * 90)
    print(
        "Response Token Length vs Uncertainty"
    )
    print("=" * 90)

    print(
        f"{'Method':<15}"
        f"{'N':>12}"
        f"{'Pearson r':>16}"
        f"{'Spearman rho':>18}"
    )

    print(
        "-" * 90
    )

    for _, row in (
        subset.iterrows()
    ):

        print(
            f"{row['method']:<15}"
            f"{int(row['n']):>12,d}"
            f"{row['pearson_r']:>16.6f}"
            f"{row['spearman_rho']:>18.6f}"
        )

    print(
        "=" * 90
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--length-column",
        type=str,
        default=(
            "response_token_length"
        ),
        choices=(
            LENGTH_COLUMNS
        ),
    )

    parser.add_argument(
        "--bin-size",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load only columns needed for this analysis
    # ========================================================

    required_columns = [
        "dataset",
        "trial",
        "problem_id",
        "candidate_idx",

        "response_char_length",
        "response_word_length",
        "response_token_length",

        "truncated_512",

        "caution_uncertainty",
        "neuboots_uncertainty",
    ]

    print(
        f"Loading: {args.input}"
    )

    df = pd.read_csv(
        args.input,
        usecols=required_columns,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Datasets: "
        f"{df['dataset'].unique().tolist()}"
    )

    print(
        f"Trials: "
        f"{sorted(df['trial'].unique())}"
    )

    # ========================================================
    # Correlation table
    # ========================================================

    correlation_df = (
        build_correlation_table(
            df
        )
    )

    correlation_path = (
        args.output_dir
        / "correlation_summary.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    # ========================================================
    # Truncation table
    # ========================================================

    truncation_df = (
        build_truncation_table(
            df=df,
            length_col=(
                args.length_column
            ),
        )
    )

    truncation_path = (
        args.output_dir
        / "truncation_correlation.csv"
    )

    truncation_df.to_csv(
        truncation_path,
        index=False,
    )

    # ========================================================
    # Length bins
    # ========================================================

    bin_df = build_bin_table(
        df=df,
        length_col=(
            args.length_column
        ),
        bin_size=args.bin_size,
    )

    bin_path = (
        args.output_dir
        / "length_bin_summary.csv"
    )

    bin_df.to_csv(
        bin_path,
        index=False,
    )

    # ========================================================
    # Graph 1: Caution density
    # ========================================================

    plot_hexbin(
        df=df,
        length_col=(
            args.length_column
        ),
        uncertainty_col=(
            "caution_uncertainty"
        ),
        method_name="Caution",
        output_path=(
            args.output_dir
            / "caution_length_uncertainty_hexbin.png"
        ),
    )

    # ========================================================
    # Graph 2: NeuBoots density
    # ========================================================

    plot_hexbin(
        df=df,
        length_col=(
            args.length_column
        ),
        uncertainty_col=(
            "neuboots_uncertainty"
        ),
        method_name="NeuBoots",
        output_path=(
            args.output_dir
            / "neuboots_length_uncertainty_hexbin.png"
        ),
    )

    # ========================================================
    # Graph 3/4: binned means
    # ========================================================

    plot_binned_curve(
        bin_df=bin_df,
        method_name="Caution",
        min_bin_count=(
            args.min_bin_count
        ),
        output_path=(
            args.output_dir
            / "caution_length_binned.png"
        ),
    )

    plot_binned_curve(
        bin_df=bin_df,
        method_name="NeuBoots",
        min_bin_count=(
            args.min_bin_count
        ),
        output_path=(
            args.output_dir
            / "neuboots_length_binned.png"
        ),
    )

    # ========================================================
    # Graph 5: standardized comparison
    # ========================================================

    plot_standardized_comparison(
        bin_df=bin_df,
        min_bin_count=(
            args.min_bin_count
        ),
        output_path=(
            args.output_dir
            / "caution_vs_neuboots_standardized.png"
        ),
    )

    # ========================================================
    # Print main result
    # ========================================================

    print_main_table(
        correlation_df
    )

    print()
    print(
        f"Saved correlation table:\n"
        f"  {correlation_path}"
    )

    print(
        f"Saved truncation table:\n"
        f"  {truncation_path}"
    )

    print(
        f"Saved length-bin table:\n"
        f"  {bin_path}"
    )

    print()
    print(
        "Plots saved in:"
    )

    print(
        f"  {args.output_dir}"
    )


if __name__ == "__main__":
    main()