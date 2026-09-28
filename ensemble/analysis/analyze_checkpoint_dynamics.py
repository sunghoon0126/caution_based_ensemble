import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_scoring_file(path):

    all_mc_mean = []
    all_uncertainty = []

    num_problems = 0
    num_candidates = 0

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            mc_mean = np.asarray(
                item[
                    "all_neuboots_mc_mean"
                ],
                dtype=np.float64,
            )

            uncertainty = np.asarray(
                item[
                    "all_neuboots_uncertainty"
                ],
                dtype=np.float64,
            )

            if (
                len(mc_mean)
                != len(uncertainty)
            ):
                raise ValueError(
                    "MC mean / uncertainty "
                    "length mismatch"
                )

            all_mc_mean.append(
                mc_mean
            )

            all_uncertainty.append(
                uncertainty
            )

            num_problems += 1

            num_candidates += (
                len(mc_mean)
            )

    all_mc_mean = np.concatenate(
        all_mc_mean
    )

    all_uncertainty = np.concatenate(
        all_uncertainty
    )

    return {
        "num_problems":
            num_problems,

        "num_candidates":
            num_candidates,

        "mc_mean_mean":
            float(
                np.mean(
                    all_mc_mean
                )
            ),

        "mc_mean_std":
            float(
                np.std(
                    all_mc_mean,
                    ddof=0,
                )
            ),

        "uncertainty_mean":
            float(
                np.mean(
                    all_uncertainty
                )
            ),

        "uncertainty_std":
            float(
                np.std(
                    all_uncertainty,
                    ddof=0,
                )
            ),

        "uncertainty_median":
            float(
                np.median(
                    all_uncertainty
                )
            ),

        "uncertainty_q25":
            float(
                np.quantile(
                    all_uncertainty,
                    0.25,
                )
            ),

        "uncertainty_q75":
            float(
                np.quantile(
                    all_uncertainty,
                    0.75,
                )
            ),

        "uncertainty_q90":
            float(
                np.quantile(
                    all_uncertainty,
                    0.90,
                )
            ),

        "uncertainty_q95":
            float(
                np.quantile(
                    all_uncertainty,
                    0.95,
                )
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--scores-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ======================================================
    # Training history
    # ======================================================

    history = pd.read_csv(
        args.history
    )

    print()
    print("Training history")
    print(history.to_string(index=False))

    # ======================================================
    # Scoring statistics
    # ======================================================

    rows = []

    for epoch in args.epochs:

        epoch_string = (
            f"{epoch:03d}"
        )

        path = os.path.join(
            args.scores_dir,
            f"epoch_{epoch_string}.jsonl",
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                path
            )

        print(
            f"\nReading epoch "
            f"{epoch_string}: "
            f"{path}"
        )

        metrics = load_scoring_file(
            path
        )

        rows.append(
            {
                "epoch":
                    epoch,

                **metrics,
            }
        )

    scoring_df = pd.DataFrame(
        rows
    )

    # ======================================================
    # Merge training + scoring
    # ======================================================

    summary = pd.merge(
        history,
        scoring_df,
        on="epoch",
        how="outer",
    )

    summary = summary.sort_values(
        "epoch"
    )

    output_csv = os.path.join(
        args.output_dir,
        "checkpoint_summary.csv",
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 120)
    print("Checkpoint summary")
    print("=" * 120)

    print(
        summary.to_string(
            index=False
        )
    )

    # ======================================================
    # Plot 1: validation loss
    # ======================================================

    if (
        "val_mc_mse"
        in summary.columns
    ):

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            summary["epoch"],
            summary["val_mc_mse"],
            marker="o",
            label="MC mean MSE",
        )

        if (
            "val_deterministic_mse"
            in summary.columns
        ):

            plt.plot(
                summary["epoch"],
                summary[
                    "val_deterministic_mse"
                ],
                marker="o",
                label="Deterministic MSE",
            )

        plt.xlabel(
            "Training epoch"
        )

        plt.ylabel(
            "Validation MSE"
        )

        plt.title(
            "Validation loss across checkpoints"
        )

        plt.legend()

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                args.output_dir,
                "validation_loss_by_epoch.png",
            ),
            dpi=200,
        )

        plt.close()

    # ======================================================
    # Plot 2: mean uncertainty
    # ======================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        summary["epoch"],
        summary["uncertainty_mean"],
        marker="o",
    )

    plt.xlabel(
        "Training epoch"
    )

    plt.ylabel(
        "Mean NeuBoots uncertainty"
    )

    plt.title(
        "Mean uncertainty across checkpoints"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            args.output_dir,
            "mean_uncertainty_by_epoch.png",
        ),
        dpi=200,
    )

    plt.close()

    # ======================================================
    # Plot 3: uncertainty distribution
    # ======================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        summary["epoch"],
        summary["uncertainty_median"],
        marker="o",
        label="Median",
    )

    plt.plot(
        summary["epoch"],
        summary["uncertainty_q25"],
        marker="o",
        label="25th percentile",
    )

    plt.plot(
        summary["epoch"],
        summary["uncertainty_q75"],
        marker="o",
        label="75th percentile",
    )

    plt.plot(
        summary["epoch"],
        summary["uncertainty_q95"],
        marker="o",
        label="95th percentile",
    )

    plt.xlabel(
        "Training epoch"
    )

    plt.ylabel(
        "NeuBoots uncertainty"
    )

    plt.title(
        "Uncertainty distribution across checkpoints"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            args.output_dir,
            "uncertainty_distribution_by_epoch.png",
        ),
        dpi=200,
    )

    plt.close()

    # ======================================================
    # Plot 4: loss vs uncertainty
    # ======================================================

    if (
        "val_mc_mse"
        in summary.columns
    ):

        valid = summary.dropna(
            subset=[
                "val_mc_mse",
                "uncertainty_mean",
            ]
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            valid["val_mc_mse"],
            valid["uncertainty_mean"],
            marker="o",
        )

        for _, row in (
            valid.iterrows()
        ):

            plt.annotate(
                f"E{int(row['epoch'])}",
                (
                    row["val_mc_mse"],
                    row[
                        "uncertainty_mean"
                    ],
                ),
            )

        plt.xlabel(
            "Validation MC MSE"
        )

        plt.ylabel(
            "Mean NeuBoots uncertainty"
        )

        plt.title(
            "Prediction error vs uncertainty "
            "during training"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                args.output_dir,
                "loss_vs_uncertainty.png",
            ),
            dpi=200,
        )

        plt.close()

    print()
    print(
        f"Saved: {output_csv}"
    )

    print(
        f"Plots: {args.output_dir}"
    )


if __name__ == "__main__":
    main()