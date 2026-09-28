import argparse
from pathlib import Path

import pandas as pd


N_VALUES = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256, 512,
]

LAMBDAS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]

TRIALS = [
    1,
    2,
    3,
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    # --------------------------------------------------------
    # Load all 3 x 6 curves
    # --------------------------------------------------------

    for trial in TRIALS:
        for lam in LAMBDAS:

            path = (
                args.input_root
                / f"trial{trial}_lambda{lam:.1f}"
                / "curves.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            df = pd.read_csv(
                path
            )

            # Only RM + NeuBoots
            df = df[
                df["method"]
                == "rm_neuboots"
            ].copy()

            df["trial"] = trial
            df["lambda"] = lam

            all_rows.append(
                df
            )

    trial_df = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 3-trial mean curve
    # --------------------------------------------------------

    aggregate_df = (
        trial_df
        .groupby(
            [
                "lambda",
                "N",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy_percent=(
                "accuracy_percent",
                "mean",
            ),
            std_accuracy_percent=(
                "accuracy_percent",
                "std",
            ),
        )
    )

    # --------------------------------------------------------
    # Peak / Final
    # --------------------------------------------------------

    summary_rows = []

    for lam in LAMBDAS:

        subset = (
            aggregate_df[
                aggregate_df["lambda"]
                == lam
            ]
            .sort_values("N")
            .reset_index(drop=True)
        )

        peak_idx = (
            subset[
                "mean_accuracy_percent"
            ]
            .idxmax()
        )

        peak_row = (
            subset.loc[
                peak_idx
            ]
        )

        final_row = (
            subset[
                subset["N"] == 512
            ]
            .iloc[0]
        )

        peak = float(
            peak_row[
                "mean_accuracy_percent"
            ]
        )

        final = float(
            final_row[
                "mean_accuracy_percent"
            ]
        )

        summary_rows.append(
            {
                "lambda":
                    lam,

                "peak_accuracy":
                    peak,

                "peak_N":
                    int(
                        peak_row["N"]
                    ),

                "final_accuracy":
                    final,

                "degradation":
                    peak - final,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trial_path = (
        args.output_dir
        / "lambda_trials.csv"
    )

    aggregate_path = (
        args.output_dir
        / "lambda_mean_curves.csv"
    )

    summary_path = (
        args.output_dir
        / "lambda_selection.csv"
    )

    trial_df.to_csv(
        trial_path,
        index=False,
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 62)
    print(
        "NeuBoots lambda selection "
        "(3-trial mean curve)"
    )
    print("=" * 62)

    print(
        f"{'lambda':>8}"
        f"{'Peak':>12}"
        f"{'Peak N':>10}"
        f"{'Final':>12}"
        f"{'Deg.':>10}"
    )

    print("-" * 62)

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['lambda']:>8.1f}"
            f"{row['peak_accuracy']:>12.2f}"
            f"{int(row['peak_N']):>10d}"
            f"{row['final_accuracy']:>12.2f}"
            f"{row['degradation']:>10.2f}"
        )

    # Select by highest peak,
    # then highest final as tiebreaker.
    selected = (
        summary_df
        .sort_values(
            [
                "peak_accuracy",
                "final_accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    print()
    print("=" * 62)
    print(
        f"Selected lambda = "
        f"{selected['lambda']:.1f}"
    )
    print(
        f"Peak           = "
        f"{selected['peak_accuracy']:.2f}%"
    )
    print(
        f"Peak N         = "
        f"{int(selected['peak_N'])}"
    )
    print(
        f"Final @512     = "
        f"{selected['final_accuracy']:.2f}%"
    )
    print("=" * 62)

    print()
    print(
        f"Saved: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()