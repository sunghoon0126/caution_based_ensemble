#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Experiment settings
# ============================================================

EPOCHS = [
    0, 1, 2, 3, 4, 5,
]

DATASETS = [
    "gsm8k",
    "math500",
    "bbh",
]

TRIALS = [
    1, 2, 3,
]


# ============================================================
# Method to compare across checkpoints
# ============================================================

METHOD = "rm_neuboots"

METHOD_DISPLAY_NAME = "RM + NeuBoots"


N_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
]


# ============================================================
# Plot setting
#
# Accuracy is already stored in percentage-point units.
#
# Example:
#
#     min = 72.4
#     max = 81.1
#
# becomes:
#
#     ylim = [69.4, 84.1]
#
# ============================================================

Y_MARGIN = 3.0


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path.home()
    / "workplace"
    / "caution_based_ensemble"
)


EVAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoint_dynamics"
    / "evaluation_all"
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoint_dynamics"
    / "epoch_comparison_plots"
)


# ============================================================
# Load one trial at one epoch
# ============================================================

def load_trial_curve(
        dataset,
        trial,
        epoch,
):
    path = (
        EVAL_ROOT
        / dataset
        / f"trial{trial}"
        / f"epoch_{epoch:03d}"
        / "curves.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation file:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    df = df[
        df["method"]
        == METHOD
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No '{METHOD}' rows found in:\n"
            f"{path}"
        )

    return df


# ============================================================
# Average trial 1 / 2 / 3
# ============================================================

def aggregate_epoch(
        dataset,
        epoch,
):
    """
    For a fixed:

        dataset
        epoch

    average the accuracy at each N across:

        trial1
        trial2
        trial3
    """

    frames = []

    for trial in TRIALS:

        df = load_trial_curve(
            dataset=dataset,
            trial=trial,
            epoch=epoch,
        )

        df = df[
            [
                "N",
                "accuracy_percent",
            ]
        ].copy()

        df["trial"] = trial

        frames.append(
            df
        )

    all_trials = pd.concat(
        frames,
        ignore_index=True,
    )

    aggregated = (
        all_trials
        .groupby(
            "N",
            as_index=False,
        )
        .agg(
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

    aggregated[
        "dataset"
    ] = dataset

    aggregated[
        "epoch"
    ] = epoch

    return aggregated


# ============================================================
# Aggregate all epochs for one dataset
# ============================================================

def aggregate_dataset(
        dataset,
):
    frames = []

    for epoch in EPOCHS:

        print(
            f"  Epoch {epoch}"
        )

        df = aggregate_epoch(
            dataset=dataset,
            epoch=epoch,
        )

        frames.append(
            df
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# Plot one dataset
# ============================================================

def plot_dataset(
        df,
        dataset,
):
    """
    One figure per dataset.

    Six N-curves are shown:

        epoch 0
        epoch 1
        ...
        epoch 5
    """

    plt.figure(
        figsize=(9, 6)
    )

    # ========================================================
    # Plot one curve per epoch
    # ========================================================

    for epoch in EPOCHS:

        subset = df[
            df["epoch"]
            == epoch
        ].copy()

        subset = (
            subset
            .sort_values(
                "N"
            )
        )

        plt.plot(
            subset["N"],
            subset[
                "accuracy_percent_mean"
            ],
            marker="o",
            linewidth=2,
            label=f"Epoch {epoch}",
        )

    # ========================================================
    # X-axis
    # ========================================================

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

    # ========================================================
    # Dynamic Y-axis
    #
    # Take min/max over every epoch and every N.
    #
    # Add ±3 percentage points.
    # ========================================================

    all_values = (
        df[
            "accuracy_percent_mean"
        ]
        .to_numpy()
    )

    data_min = float(
        np.min(
            all_values
        )
    )

    data_max = float(
        np.max(
            all_values
        )
    )

    y_min = (
        data_min
        - Y_MARGIN
    )

    y_max = (
        data_max
        + Y_MARGIN
    )

    # Accuracy cannot meaningfully exceed 0~100.
    y_min = max(
        0.0,
        y_min,
    )

    y_max = min(
        100.0,
        y_max,
    )

    plt.ylim(
        y_min,
        y_max,
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    # ========================================================
    # Plot style
    # ========================================================

    plt.title(
        f"{dataset.upper()} - "
        f"{METHOD_DISPLAY_NAME} "
        "Across Checkpoints"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend(
        title="Checkpoint",
    )

    plt.tight_layout()

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / f"{dataset}_epochs_0_5.png"
    )

    plt.savefig(
        output_path,
        dpi=250,
    )

    plt.close()

    print(
        f"  Data range : "
        f"{data_min:.2f}% ~ {data_max:.2f}%"
    )

    print(
        f"  Y-axis     : "
        f"{y_min:.2f}% ~ {y_max:.2f}%"
    )

    print(
        f"  Saved      : "
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_frames = []

    for dataset in DATASETS:

        print()
        print(
            "=" * 70
        )

        print(
            f"Dataset: {dataset.upper()}"
        )

        print(
            "=" * 70
        )

        dataset_df = aggregate_dataset(
            dataset=dataset,
        )

        all_frames.append(
            dataset_df
        )

        plot_dataset(
            df=dataset_df,
            dataset=dataset,
        )

    # ========================================================
    # Save numerical values used for the figures
    # ========================================================

    all_df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    csv_path = (
        OUTPUT_ROOT
        / "trial_averaged_epoch_curves.csv"
    )

    all_df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Finished"
    )

    print(
        "=" * 70
    )

    print(
        "Saved curves:"
    )

    print(
        f"  {csv_path}"
    )


if __name__ == "__main__":
    main()