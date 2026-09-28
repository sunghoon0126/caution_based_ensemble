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

EPOCHS = [
    0,
    1,
    2,
    3,
    4,
    5,
]

METHODS = [
    "rm",
    "neuboots_pessimism",
    "rm_neuboots",
]


def load_all_curves(
    evaluation_root: Path,
):
    rows = []

    for dataset in DATASETS:

        for trial in TRIALS:

            for epoch in EPOCHS:

                epoch_name = (
                    f"epoch_{epoch:03d}"
                )

                path = (
                    evaluation_root
                    / dataset
                    / f"trial{trial}"
                    / epoch_name
                    / "curves.csv"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing curve file: {path}"
                    )

                df = pd.read_csv(path)

                df["trial"] = trial
                df["epoch"] = epoch

                rows.append(df)

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    return result


def average_trials(
    all_curves: pd.DataFrame,
):
    """
    IMPORTANT:

    First average trial1/2/3 at each:

        dataset
        epoch
        method
        N

    Then peak/final metrics are computed
    from this averaged curve.

    We do NOT average trial-wise peaks.
    """

    grouped = (
        all_curves
        .groupby(
            [
                "dataset",
                "epoch",
                "method",
                "method_display_name",
                "N",
            ],
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
            num_trials=(
                "trial",
                "nunique",
            ),
        )
    )

    grouped[
        "accuracy_percent_sem"
    ] = (
        grouped[
            "accuracy_percent_std"
        ]
        / np.sqrt(
            grouped["num_trials"]
        )
    )

    return grouped


def build_summary(
    mean_curves: pd.DataFrame,
):
    rows = []

    for (
        dataset,
        epoch,
        method,
    ), group in mean_curves.groupby(
        [
            "dataset",
            "epoch",
            "method",
        ]
    ):

        group = (
            group
            .sort_values("N")
            .reset_index(drop=True)
        )

        peak_index = (
            group[
                "accuracy_percent_mean"
            ]
            .idxmax()
        )

        peak_row = (
            group.loc[peak_index]
        )

        final_row = (
            group.iloc[-1]
        )

        peak_accuracy = float(
            peak_row[
                "accuracy_percent_mean"
            ]
        )

        final_accuracy = float(
            final_row[
                "accuracy_percent_mean"
            ]
        )

        rows.append(
            {
                "dataset":
                    dataset,

                "epoch":
                    int(epoch),

                "method":
                    method,

                "peak_n":
                    int(
                        peak_row["N"]
                    ),

                "peak_accuracy":
                    peak_accuracy,

                "final_n":
                    int(
                        final_row["N"]
                    ),

                "final_accuracy":
                    final_accuracy,

                "degradation":
                    (
                        peak_accuracy
                        - final_accuracy
                    ),
            }
        )

    return pd.DataFrame(rows)


def load_training_history(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training history: {path}"
        )

    history = pd.read_csv(path)

    keep_columns = [
        "epoch",
        "global_step",
        "train_loss",
        "val_mc_mse",
        "val_mae",
        "val_uncertainty_mean",
    ]

    keep_columns = [
        c
        for c in keep_columns
        if c in history.columns
    ]

    return history[keep_columns]


def plot_bon_curves(
    mean_curves: pd.DataFrame,
    output_dir: Path,
):
    """
    One plot for each dataset,
    showing epoch 0~5 curves for RM+NeuBoots.
    """

    method = "rm_neuboots"

    for dataset in DATASETS:

        subset = mean_curves[
            (
                mean_curves["dataset"]
                == dataset
            )
            &
            (
                mean_curves["method"]
                == method
            )
        ]

        plt.figure(
            figsize=(8, 5.5)
        )

        for epoch in EPOCHS:

            epoch_df = (
                subset[
                    subset["epoch"]
                    == epoch
                ]
                .sort_values("N")
            )

            plt.plot(
                epoch_df["N"],
                epoch_df[
                    "accuracy_percent_mean"
                ],
                marker="o",
                label=f"Epoch {epoch}",
            )

        plt.xscale("log", base=2)

        plt.xlabel(
            "Best-of-N"
        )

        plt.ylabel(
            "Accuracy (%)"
        )

        plt.title(
            f"{dataset.upper()} - "
            "RM + NeuBoots"
        )

        plt.xticks(
            [
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
            ],
            [
                "1",
                "2",
                "4",
                "8",
                "16",
                "32",
                "64",
                "128",
                "256",
                "512",
            ],
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        path = (
            output_dir
            / (
                f"{dataset}_"
                "rm_neuboots_"
                "epochs.png"
            )
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()


def plot_epoch_summary(
    summary: pd.DataFrame,
    output_dir: Path,
):
    """
    Dataset-wise:
        epoch -> peak accuracy
        epoch -> final N=512 accuracy
        epoch -> degradation
    """

    subset = summary[
        summary["method"]
        == "rm_neuboots"
    ]

    for metric, ylabel in [
        (
            "peak_accuracy",
            "Peak accuracy (%)",
        ),
        (
            "final_accuracy",
            "Accuracy at N=512 (%)",
        ),
        (
            "degradation",
            "Peak - final accuracy (%)",
        ),
    ]:

        plt.figure(
            figsize=(8, 5.5)
        )

        for dataset in DATASETS:

            dataset_df = (
                subset[
                    subset["dataset"]
                    == dataset
                ]
                .sort_values("epoch")
            )

            plt.plot(
                dataset_df["epoch"],
                dataset_df[metric],
                marker="o",
                label=dataset.upper(),
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)

        plt.title(
            f"Checkpoint dynamics - {ylabel}"
        )

        plt.xticks(EPOCHS)

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"epoch_vs_{metric}.png",
            dpi=200,
        )

        plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path(
            "outputs/checkpoint_dynamics/"
            "evaluation_all"
        ),
    )

    parser.add_argument(
        "--training-history",
        type=Path,
        default=Path(
            "outputs/checkpoint_dynamics/"
            "gsm8k_lightweight/"
            "training_history.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/checkpoint_dynamics/"
            "aggregate_all"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load all 54 evaluations
    # --------------------------------------------------------

    all_curves = load_all_curves(
        args.evaluation_root
    )

    all_curves.to_csv(
        args.output_dir
        / "all_trial_curves.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Trial average
    # --------------------------------------------------------

    mean_curves = average_trials(
        all_curves
    )

    mean_curves.to_csv(
        args.output_dir
        / "mean_curves_by_epoch.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Peak / final / degradation
    # --------------------------------------------------------

    summary = build_summary(
        mean_curves
    )

    # --------------------------------------------------------
    # 4. Add training dynamics
    #
    # Same GSM8K-trained predictor is evaluated
    # on all three datasets, so training metrics
    # are shared across datasets.
    # --------------------------------------------------------

    history = load_training_history(
        args.training_history
    )

    summary = summary.merge(
        history,
        on="epoch",
        how="left",
    )

    summary.to_csv(
        args.output_dir
        / "table3_checkpoint_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Compact RM+NeuBoots table
    # --------------------------------------------------------

    compact = (
        summary[
            summary["method"]
            == "rm_neuboots"
        ]
        .sort_values(
            [
                "dataset",
                "epoch",
            ]
        )
        .reset_index(drop=True)
    )

    compact.to_csv(
        args.output_dir
        / "rm_neuboots_checkpoint_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Plots
    # --------------------------------------------------------

    plot_bon_curves(
        mean_curves,
        args.output_dir,
    )

    plot_epoch_summary(
        summary,
        args.output_dir,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "RM + NeuBoots checkpoint summary"
    )
    print("=" * 120)

    display_columns = [
        "dataset",
        "epoch",
        "train_loss",
        "val_mc_mse",
        "peak_n",
        "peak_accuracy",
        "final_accuracy",
        "degradation",
    ]

    print(
        compact[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()