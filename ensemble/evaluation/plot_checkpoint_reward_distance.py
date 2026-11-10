#!/usr/bin/env python3

import json
import re
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

EPOCHS = [
    0,
    1,
    2,
    3,
    4,
    5,
]

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


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path.home()
    / "workplace"
    / "caution_based_ensemble"
)

CAUTION_ROOT = (
    Path.home()
    / "workplace"
    / "Caution"
)

CAUTION_EVAL_ROOT = (
    CAUTION_ROOT
    / "outputs"
    / "repro_v2"
    / "repro_v2_eval_20260825_173851"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoint_dynamics"
)

OUTPUT_ROOT = (
    CHECKPOINT_ROOT
    / "reward_distance"
)


# ============================================================
# Candidate ordering
# ============================================================

def candidate_key(value):

    text = str(value)

    numbers = re.findall(
        r"\d+",
        text,
    )

    if numbers:
        return (
            int(numbers[-1]),
            text,
        )

    return (
        10**12,
        text,
    )


# ============================================================
# Stream top-level JSON
# ============================================================

def iter_top_level(path):

    with open(
        path,
        "rb",
    ) as f:

        first = f.read(1)

        while first in {
            b" ",
            b"\n",
            b"\r",
            b"\t",
        }:
            first = f.read(1)

        f.seek(0)

        if first == b"{":

            yield from ijson.kvitems(
                f,
                "",
            )

        elif first == b"[":

            for i, item in enumerate(
                ijson.items(
                    f,
                    "item",
                )
            ):

                yield (
                    str(i),
                    item,
                )

        else:

            raise ValueError(
                f"Unsupported JSON root: {path}"
            )


# ============================================================
# Find Caution detailed candidate file
# ============================================================

def find_caution_detail(
        dataset,
        trial,
):

    directory = (
        CAUTION_EVAL_ROOT
        / dataset
        / f"trial{trial}"
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Missing directory:\n{directory}"
        )

    matches = list(
        directory.glob(
            "*detailed_candidates.json"
        )
    )

    if len(matches) == 0:

        raise RuntimeError(
            "No detailed_candidates.json found in:\n"
            f"{directory}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            "Multiple detailed candidate files found in:\n"
            f"{directory}\n"
            f"{matches}"
        )

    return matches[0]


# ============================================================
# Existing checkpoint score path
# ============================================================

def get_score_path(
        dataset,
        trial,
        epoch,
):
    """
    Historical checkpoint experiment used
    a separate location for GSM8K trial1.
    """

    epoch_name = (
        f"epoch_{epoch:03d}.jsonl"
    )

    if (
        dataset == "gsm8k"
        and trial == 1
    ):

        return (
            CHECKPOINT_ROOT
            / "gsm8k_trial1_scores"
            / epoch_name
        )

    return (
        CHECKPOINT_ROOT
        / "scored"
        / dataset
        / f"trial{trial}"
        / epoch_name
    )


# ============================================================
# Load main RM rewards
# ============================================================

def load_rm_scores(path):

    data = {}

    for (
        problem_id,
        problem,
    ) in iter_top_level(path):

        responses = (
            problem[
                "responses"
            ]
        )

        if isinstance(
            responses,
            dict,
        ):

            candidates = [
                responses[key]
                for key in sorted(
                    responses,
                    key=candidate_key,
                )
            ]

        else:

            candidates = responses

        rm_values = []

        for candidate in candidates:

            scores = candidate.get(
                "reward_scores",
                {},
            )

            rm = scores.get(
                "reward_model_score",
                candidate.get(
                    "reward_model_score"
                ),
            )

            if rm is None:

                raise RuntimeError(
                    "Missing reward model score "
                    f"for problem {problem_id}"
                )

            rm_values.append(
                float(rm)
            )

        data[
            str(problem_id)
        ] = np.asarray(
            rm_values,
            dtype=np.float64,
        )

    return data


# ============================================================
# Load NeuBoots ensemble mean
# ============================================================

def load_ensemble_means(path):

    data = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(
                line
            )

            problem_id = str(
                item[
                    "instance_id"
                ]
            )

            mc_mean = np.asarray(
                item[
                    "all_neuboots_mc_mean"
                ],
                dtype=np.float64,
            )

            data[
                problem_id
            ] = mc_mean

    return data


# ============================================================
# Compute one trial / one epoch distance
# ============================================================

def compute_trial_distance(
        dataset,
        trial,
        epoch,
):

    detail_path = (
        find_caution_detail(
            dataset=dataset,
            trial=trial,
        )
    )

    score_path = (
        get_score_path(
            dataset=dataset,
            trial=trial,
            epoch=epoch,
        )
    )

    if not score_path.exists():

        raise FileNotFoundError(
            f"Missing NeuBoots score file:\n"
            f"{score_path}"
        )

    rm_data = load_rm_scores(
        detail_path
    )

    ensemble_data = (
        load_ensemble_means(
            score_path
        )
    )

    # ========================================================
    # ID alignment
    # ========================================================

    rm_ids = set(
        rm_data
    )

    ensemble_ids = set(
        ensemble_data
    )

    if rm_ids != ensemble_ids:

        missing = (
            rm_ids
            - ensemble_ids
        )

        extra = (
            ensemble_ids
            - rm_ids
        )

        raise RuntimeError(
            "RM / NeuBoots IDs do not match.\n"
            f"Missing: {len(missing)}\n"
            f"Extra: {len(extra)}"
        )

    # ========================================================
    # Candidate-level:
    #
    # | main RM reward - ensemble mean reward |
    # ========================================================

    all_distances = []

    for problem_id in rm_data:

        rm_values = (
            rm_data[
                problem_id
            ]
        )

        ensemble_values = (
            ensemble_data[
                problem_id
            ]
        )

        if (
            len(rm_values)
            != len(ensemble_values)
        ):

            raise RuntimeError(
                "Candidate count mismatch "
                f"for {problem_id}: "
                f"RM={len(rm_values)}, "
                f"NeuBoots={len(ensemble_values)}"
            )

        distances = np.abs(
            rm_values
            - ensemble_values
        )

        all_distances.append(
            distances
        )

    pooled = np.concatenate(
        all_distances
    )

    return {
        "dataset":
            dataset,

        "trial":
            trial,

        "epoch":
            epoch,

        "num_candidates":
            int(
                pooled.size
            ),

        "mean_abs_distance":
            float(
                pooled.mean()
            ),

        "median_abs_distance":
            float(
                np.median(
                    pooled
                )
            ),
    }


# ============================================================
# Plot one dataset
# ============================================================

def plot_dataset(
        epoch_df,
        dataset,
):

    subset = epoch_df[
        epoch_df[
            "dataset"
        ] == dataset
    ].copy()

    subset = (
        subset
        .sort_values(
            "epoch"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Values
    # ========================================================

    values = (
        subset[
            "mean_abs_distance"
        ]
        .to_numpy()
    )

    data_min = float(
        np.min(values)
    )

    data_max = float(
        np.max(values)
    )

    # ========================================================
    # Dynamic Y-axis
    #
    # margin = 3% of the observed value range
    # ========================================================

    value_range = (
        data_max
        - data_min
    )

    margin = (
        value_range
        * 0.03
    )

    # If all epochs are almost identical,
    # still give the plot some visible space.
    if margin <= 1e-12:

        margin = max(
            abs(data_max)
            * 0.03,
            1e-4,
        )

    y_min = (
        data_min
        - margin
    )

    y_max = (
        data_max
        + margin
    )

    # Absolute distance cannot be negative.
    y_min = max(
        0.0,
        y_min,
    )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.plot(
        subset["epoch"],
        subset[
            "mean_abs_distance"
        ],
        marker="o",
        linewidth=2,
    )

    # Optional error bars representing
    # variation across the three trials.
    plt.errorbar(
        subset["epoch"],
        subset[
            "mean_abs_distance"
        ],
        yerr=subset[
            "std_across_trials"
        ],
        fmt="none",
        capsize=4,
        alpha=0.6,
    )

    # ========================================================
    # Axes
    # ========================================================

    plt.xticks(
        EPOCHS
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mean |RM Reward - Ensemble Mean Reward|"
    )

    plt.ylim(
        y_min,
        y_max,
    )

    # ========================================================
    # Style
    # ========================================================

    plt.title(
        f"{dataset.upper()} - "
        "RM vs Ensemble Mean Reward"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    # ========================================================
    # Save
    # ========================================================

    output_path = (
        OUTPUT_ROOT
        / f"{dataset}_epoch_reward_distance.png"
    )

    plt.savefig(
        output_path,
        dpi=250,
    )

    plt.close()

    print()
    print(
        f"{dataset.upper()}"
    )

    print(
        f"  Data range : "
        f"{data_min:.6f} ~ "
        f"{data_max:.6f}"
    )

    print(
        f"  Margin     : "
        f"{margin:.6f}"
    )

    print(
        f"  Y-axis     : "
        f"{y_min:.6f} ~ "
        f"{y_max:.6f}"
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

    rows = []

    # ========================================================
    # dataset x epoch x trial
    # ========================================================

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

        for epoch in EPOCHS:

            for trial in TRIALS:

                print(
                    f"epoch={epoch}, "
                    f"trial={trial}"
                )

                result = (
                    compute_trial_distance(
                        dataset=dataset,
                        trial=trial,
                        epoch=epoch,
                    )
                )

                rows.append(
                    result
                )

    # ========================================================
    # Save individual trial results
    # ========================================================

    trial_df = pd.DataFrame(
        rows
    )

    trial_path = (
        OUTPUT_ROOT
        / "reward_distance_by_trial.csv"
    )

    trial_df.to_csv(
        trial_path,
        index=False,
    )

    # ========================================================
    # Trial average
    #
    # For every dataset and epoch:
    #
    #   (trial1 + trial2 + trial3) / 3
    # ========================================================

    epoch_df = (
        trial_df
        .groupby(
            [
                "dataset",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_distance=(
                "mean_abs_distance",
                "mean",
            ),

            std_across_trials=(
                "mean_abs_distance",
                "std",
            ),
        )
        .sort_values(
            [
                "dataset",
                "epoch",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    epoch_path = (
        OUTPUT_ROOT
        / "reward_distance_by_epoch.csv"
    )

    epoch_df.to_csv(
        epoch_path,
        index=False,
    )

    # ========================================================
    # Print values
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "Trial-averaged reward distance"
    )

    print(
        "=" * 80
    )

    print(
        epoch_df.to_string(
            index=False
        )
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Three figures
    # ========================================================

    for dataset in DATASETS:

        plot_dataset(
            epoch_df=epoch_df,
            dataset=dataset,
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
        "Trial-level values:"
    )

    print(
        f"  {trial_path}"
    )

    print(
        "Epoch-level values:"
    )

    print(
        f"  {epoch_path}"
    )


if __name__ == "__main__":
    main()