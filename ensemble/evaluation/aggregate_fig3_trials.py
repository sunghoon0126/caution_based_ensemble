#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Methods used in the NeuBoots version of Figure 3
# ============================================================

METHODS = [
    "rm",
    "rm_neuboots",
    "neuboots_pessimism",
]


# ============================================================
# Dataset-name normalization
#
# plot_fig3_methods.py expects:
#
#     GSM8K
#     MATH-500
#     BBH
#
# while evaluate_fig3_methods.py may save:
#
#     gsm8k
#     math500
#     bbh
# ============================================================

DATASET_NAME_MAP = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "math_500": "MATH-500",
    "math-500": "MATH-500",
    "math_500_test": "MATH-500",
    "bbh": "BBH",
}


def normalize_dataset_names(series):
    original = series.astype(str)

    normalized = (
        original
        .str.strip()
        .str.lower()
    )

    mapped = normalized.map(
        DATASET_NAME_MAP
    )

    return mapped.fillna(
        original
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate three Figure 3 evaluation "
            "trials into a mean curve with "
            "trial-to-trial standard deviation."
        )
    )

    parser.add_argument(
        "--trial-files",
        nargs=3,
        type=Path,
        required=True,
        help=(
            "Exactly three curves.csv files "
            "from trial1, trial2, and trial3."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output CSV containing the "
            "three-trial mean curves."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load trial curves
    # ========================================================

    trial_frames = []

    for trial_idx, path in enumerate(
        args.trial_files,
        start=1,
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Trial file not found: {path}"
            )

        print(
            f"Loading trial {trial_idx}: "
            f"{path}"
        )

        df = pd.read_csv(
            path
        )

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = {
            "dataset",
            "method",
            "method_display_name",
            "N",
            "accuracy_percent",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{path}: missing columns "
                f"{sorted(missing_columns)}"
            )

        # ----------------------------------------------------
        # Keep only Figure 3 NeuBoots methods
        # ----------------------------------------------------

        df = df[
            df["method"].isin(
                METHODS
            )
        ].copy()

        if df.empty:
            raise RuntimeError(
                f"No requested methods found "
                f"in {path}"
            )

        df["trial"] = (
            trial_idx
        )

        trial_frames.append(
            df
        )

    all_df = pd.concat(
        trial_frames,
        ignore_index=True,
    )

    # ========================================================
    # 2. Basic validation
    # ========================================================

    print()
    print(
        "Methods found:",
        sorted(
            all_df[
                "method"
            ].unique()
        ),
    )

    print(
        "Dataset values before normalization:",
        sorted(
            all_df[
                "dataset"
            ]
            .astype(str)
            .unique()
        ),
    )

    # ========================================================
    # 3. Normalize dataset names
    # ========================================================

    all_df["dataset"] = (
        normalize_dataset_names(
            all_df["dataset"]
        )
    )

    print(
        "Dataset values after normalization:",
        sorted(
            all_df[
                "dataset"
            ]
            .astype(str)
            .unique()
        ),
    )

    # ========================================================
    # 4. Aggregate over three trials
    #
    # For each method and N:
    #
    #     mean accuracy
    #     std across three trials
    #
    # ========================================================

    result = (
        all_df
        .groupby(
            [
                "dataset",
                "method",
                "method_display_name",
                "N",
            ],
            as_index=False,
        )
        .agg(
            accuracy_percent=(
                "accuracy_percent",
                "mean",
            ),
            std_accuracy_percent=(
                "accuracy_percent",
                "std",
            ),
            num_trials=(
                "trial",
                "nunique",
            ),
        )
    )

    # ========================================================
    # 5. Validate all points use three trials
    # ========================================================

    bad_rows = result[
        result["num_trials"] != 3
    ]

    if not bad_rows.empty:

        print()
        print(
            "Rows without exactly "
            "three trials:"
        )

        print(
            bad_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Some Figure 3 points do not "
            "contain all three trials."
        )

    # ========================================================
    # 6. Build plotting interval
    #
    # Existing plot_fig3_methods.py expects:
    #
    #     ci_low
    #     ci_high
    #
    # For the 3-trial reproduction we use:
    #
    #     mean ± 1 std
    #
    # rather than problem-level bootstrap CI.
    # ========================================================

    result["ci_low"] = (
        result[
            "accuracy_percent"
        ]
        - result[
            "std_accuracy_percent"
        ]
    )

    result["ci_high"] = (
        result[
            "accuracy_percent"
        ]
        + result[
            "std_accuracy_percent"
        ]
    )

    # Keep the original fractional accuracy
    # column expected by some evaluation scripts.
    result["accuracy"] = (
        result[
            "accuracy_percent"
        ]
        / 100.0
    )

    # ========================================================
    # 7. Sort Figure 3 curves
    # ========================================================

    method_order = {
        "rm": 0,
        "rm_neuboots": 1,
        "neuboots_pessimism": 2,
    }

    result["_method_order"] = (
        result["method"]
        .map(
            method_order
        )
    )

    result = (
        result
        .sort_values(
            [
                "dataset",
                "_method_order",
                "N",
            ]
        )
        .drop(
            columns=[
                "_method_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # 8. Save
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    # ========================================================
    # 9. Print result
    # ========================================================

    print()
    print(
        "=" * 80
    )
    print(
        "Three-trial Figure 3 "
        "aggregate"
    )
    print(
        "=" * 80
    )

    display_columns = [
        "dataset",
        "method",
        "N",
        "accuracy_percent",
        "std_accuracy_percent",
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Rows:",
        len(result),
    )

    print(
        "Methods:",
        result[
            "method"
        ].unique().tolist(),
    )

    print(
        "Datasets:",
        result[
            "dataset"
        ].unique().tolist(),
    )

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()