#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import ijson
import numpy as np
import pandas as pd

from ensemble.evaluation.normalization import (
    load_normalization_stats,
    normalize_with_stats,
)


# ============================================================
# Figure 3 evaluation grid
# ============================================================

N_VALUES = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256, 512,
]


# ============================================================
# NeuBoots uncertainty settings
# ============================================================

NEUBOOTS_UNCERTAINTY_TYPES = [
    "std",
    "distance",
    "median_quantile",
    "avg_quantile",
    "avg_quantile_avg",
]


NEUBOOTS_ALPHAS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
]


# ============================================================
# Supported methods
# ============================================================

SUPPORTED_METHODS = [
    "rm",
    "pessimism",
    "caution",
    "neuboots_pessimism",
    "rm_neuboots",
]


METHOD_DISPLAY_NAMES = {
    "rm":
        "Reward Model",

    "pessimism":
        "RND Pessimism",

    "caution":
        "RM + RND Pessimism",

    "neuboots_pessimism":
        "NeuBoots Pessimism",

    "rm_neuboots":
        "RM + NeuBoots",
}


# ============================================================
# Caution detailed candidate loader
# ============================================================

def candidate_key(value):
    """
    Preserve candidate ordering from the
    previous Figure 3 reproduction code.
    """

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


def iter_top_level(path):
    """
    Stream a top-level JSON dictionary or list.
    """

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
                f"Unsupported JSON root: "
                f"{path}"
            )


def to_accuracy(value):

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        str,
    ):

        value = (
            value
            .strip()
            .lower()
        )

        if value in {
            "true",
            "correct",
        }:
            return 1.0

        if value in {
            "false",
            "incorrect",
        }:
            return 0.0

        return float(
            value
        )

    raise TypeError(
        "Unsupported accuracy value: "
        f"{value}"
    )


def load_caution_dataset(path):
    """
    Load candidate-level:

        accuracy
        reward_model_score
        rnd_score

    Stored Caution convention:

        rnd_score = - uncertainty

    Output arrays:

        [num_problems, num_candidates]
    """

    problem_ids = []

    accuracy_rows = []
    rm_rows = []
    rnd_rows = []

    candidate_counts = []

    print(
        f"Reading Caution candidates: "
        f"{path}"
    )

    for problem_idx, (
        problem_id,
        problem,
    ) in enumerate(
        iter_top_level(
            path
        ),
        start=1,
    ):

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

        accuracy_values = []
        rm_values = []
        rnd_values = []

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

            rnd = scores.get(
                "rnd_score",
                candidate.get(
                    "rnd_score"
                ),
            )

            accuracy = candidate.get(
                "accuracy",
                candidate.get(
                    "correct",
                    candidate.get(
                        "is_correct"
                    ),
                ),
            )

            if (
                rm is None
                or rnd is None
                or accuracy is None
            ):
                raise RuntimeError(
                    "Missing RM / RND / accuracy "
                    f"field in problem "
                    f"{problem_idx}"
                )

            rm_values.append(
                float(
                    rm
                )
            )

            rnd_values.append(
                float(
                    rnd
                )
            )

            accuracy_values.append(
                to_accuracy(
                    accuracy
                )
            )

        problem_ids.append(
            str(
                problem_id
            )
        )

        candidate_counts.append(
            len(
                candidates
            )
        )

        rm_rows.append(
            np.asarray(
                rm_values,
                dtype=np.float64,
            )
        )

        rnd_rows.append(
            np.asarray(
                rnd_values,
                dtype=np.float64,
            )
        )

        accuracy_rows.append(
            np.asarray(
                accuracy_values,
                dtype=np.float64,
            )
        )

        if (
            problem_idx
            % 100
            == 0
        ):
            print(
                f"  loaded "
                f"{problem_idx} "
                f"problems"
            )

    if not candidate_counts:
        raise RuntimeError(
            "No problems were loaded."
        )

    min_candidates = min(
        candidate_counts
    )

    if (
        len(
            set(
                candidate_counts
            )
        )
        != 1
    ):
        print(
            "Warning: candidate counts differ. "
            f"Truncating to "
            f"{min_candidates}."
        )

    accuracy = np.stack(
        [
            row[
                :min_candidates
            ]
            for row
            in accuracy_rows
        ]
    )

    rm = np.stack(
        [
            row[
                :min_candidates
            ]
            for row
            in rm_rows
        ]
    )

    rnd = np.stack(
        [
            row[
                :min_candidates
            ]
            for row
            in rnd_rows
        ]
    )

    print(
        f"Problems="
        f"{accuracy.shape[0]}, "
        f"Candidates="
        f"{accuracy.shape[1]}"
    )

    return (
        problem_ids,
        accuracy,
        rm,
        rnd,
    )


# ============================================================
# NeuBoots uncertainty helpers
# ============================================================

def get_neuboots_normalization_key(
        uncertainty_type,
        alpha=None,
):
    """
    Map uncertainty configuration to the
    independently fitted normalization key.
    """

    if uncertainty_type == "std":
        return (
            "neuboots_std"
        )

    if uncertainty_type == "distance":
        return (
            "neuboots_distance"
        )

    if alpha is None:
        raise ValueError(
            "--neuboots-alpha is required "
            f"for {uncertainty_type}."
        )

    alpha_key = (
        f"{alpha:.1f}"
    )

    if (
        uncertainty_type
        == "median_quantile"
    ):
        return (
            "neuboots_median_quantile_"
            f"{alpha_key}"
        )

    if (
        uncertainty_type
        == "avg_quantile"
    ):
        return (
            "neuboots_avg_quantile_"
            f"{alpha_key}"
        )

    if (
        uncertainty_type
        == "avg_quantile_avg"
    ):
        return (
            "neuboots_avg_quantile_avg_"
            f"{alpha_key}"
        )

    raise ValueError(
        "Unsupported NeuBoots "
        "uncertainty type: "
        f"{uncertainty_type}"
    )


def get_neuboots_result_field(
        item,
        uncertainty_type,
        alpha=None,
):
    """
    Extract the selected uncertainty-related
    candidate array from one NeuBoots JSONL row.

    distance is special:
        the JSONL stores ensemble mean,
        then |RM - ensemble_mean| is computed
        after alignment with the Caution RM scores.
    """

    # --------------------------------------------------------
    # 1. STD
    # --------------------------------------------------------

    if uncertainty_type == "std":

        return item[
            "all_neuboots_uncertainty_std"
        ]

    # --------------------------------------------------------
    # 2. Distance
    #
    # Return ensemble mean here.
    # Actual distance is computed after alignment:
    #
    #     |RM - ensemble_mean|
    # --------------------------------------------------------

    if uncertainty_type == "distance":

        return item[
            "all_neuboots_mc_mean"
        ]

    # --------------------------------------------------------
    # Quantile types require alpha
    # --------------------------------------------------------

    if alpha is None:
        raise ValueError(
            "--neuboots-alpha is required "
            f"for {uncertainty_type}."
        )

    alpha_key = (
        f"{alpha:.1f}"
    )

    # --------------------------------------------------------
    # 3. Median - Quantile
    # --------------------------------------------------------

    if (
        uncertainty_type
        == "median_quantile"
    ):

        return item[
            "all_neuboots_uncertainty_median_quantile"
        ][
            alpha_key
        ]

    # --------------------------------------------------------
    # 4. Average - Quantile
    # --------------------------------------------------------

    if (
        uncertainty_type
        == "avg_quantile"
    ):

        return item[
            "all_neuboots_uncertainty_avg_quantile"
        ][
            alpha_key
        ]

    # --------------------------------------------------------
    # 5. Average - Quantile's Average
    # --------------------------------------------------------

    if (
        uncertainty_type
        == "avg_quantile_avg"
    ):

        return item[
            "all_neuboots_uncertainty_avg_quantile_avg"
        ][
            alpha_key
        ]

    raise ValueError(
        "Unsupported NeuBoots "
        "uncertainty type: "
        f"{uncertainty_type}"
    )


def get_neuboots_uncertainty_definition(
        uncertainty_type,
        alpha=None,
):
    """
    Human-readable formula for result metadata.
    """

    if uncertainty_type == "std":
        return (
            "population_std("
            "NeuBoots_MC_rewards)"
        )

    if uncertainty_type == "distance":
        return (
            "abs("
            "main_reward_model_score "
            "- NeuBoots_MC_mean"
            ")"
        )

    alpha_text = (
        "floor(alpha * M)"
    )

    if (
        uncertainty_type
        == "median_quantile"
    ):
        return (
            "sorted_MC[floor(M/2)] "
            "- sorted_MC["
            f"{alpha_text}"
            "]"
        )

    if (
        uncertainty_type
        == "avg_quantile"
    ):
        return (
            "NeuBoots_MC_mean "
            "- sorted_MC["
            f"{alpha_text}"
            "]"
        )

    if (
        uncertainty_type
        == "avg_quantile_avg"
    ):
        return (
            "NeuBoots_MC_mean "
            "- mean("
            "sorted_MC[0:"
            "floor(alpha*M)+1]"
            ")"
        )

    raise ValueError(
        "Unsupported NeuBoots "
        "uncertainty type: "
        f"{uncertainty_type}"
    )


# ============================================================
# NeuBoots result loader
# ============================================================

def load_neuboots_results(
        path,
        problem_ids,
        num_candidates,
        uncertainty_type,
        rm,
        alpha=None,
):
    """
    Load selected NeuBoots uncertainty.

    Candidate ordering must match the
    Caution candidate ordering.

    For distance:

        uncertainty =
            |RM - NeuBoots ensemble mean|

    For all other metrics, uncertainty is
    stored directly in the scoring JSONL.
    """

    data = {}

    print(
        f"Reading NeuBoots scores: "
        f"{path}"
    )

    print(
        "NeuBoots uncertainty type: "
        f"{uncertainty_type}"
    )

    if alpha is not None:
        print(
            "NeuBoots alpha: "
            f"{alpha:.1f}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = (
                line.strip()
            )

            if not line:
                continue

            item = json.loads(
                line
            )

            instance_id = str(
                item[
                    "instance_id"
                ]
            )

            if instance_id in data:
                raise ValueError(
                    "Duplicate NeuBoots "
                    "instance_id: "
                    f"{instance_id}"
                )

            values = (
                get_neuboots_result_field(
                    item=item,
                    uncertainty_type=(
                        uncertainty_type
                    ),
                    alpha=alpha,
                )
            )

            data[
                instance_id
            ] = np.asarray(
                values,
                dtype=np.float64,
            )

    # ========================================================
    # ID alignment
    # ========================================================

    caution_ids = set(
        problem_ids
    )

    neuboots_ids = set(
        data
    )

    if (
        caution_ids
        != neuboots_ids
    ):

        missing = (
            caution_ids
            - neuboots_ids
        )

        extra = (
            neuboots_ids
            - caution_ids
        )

        print()

        print(
            "Caution / NeuBoots IDs "
            "do not match."
        )

        print(
            "Missing NeuBoots IDs: "
            f"{len(missing)}"
        )

        print(
            "Extra NeuBoots IDs: "
            f"{len(extra)}"
        )

        if missing:
            print(
                "Example missing IDs:",
                list(
                    missing
                )[:5],
            )

        if extra:
            print(
                "Example extra IDs:",
                list(
                    extra
                )[:5],
            )

        raise ValueError(
            "Cannot safely align "
            "NeuBoots candidates."
        )

    # ========================================================
    # Candidate alignment
    # ========================================================

    rows = []

    for (
        problem_idx,
        problem_id,
    ) in enumerate(
        problem_ids
    ):

        values = data[
            problem_id
        ]

        if (
            len(values)
            < num_candidates
        ):
            raise ValueError(
                f"{problem_id}: "
                "NeuBoots has only "
                f"{len(values)} candidates, "
                "expected at least "
                f"{num_candidates}."
            )

        values = values[
            :num_candidates
        ]

        # ----------------------------------------------------
        # 2. Distance
        #
        # values currently contains:
        #
        #     NeuBoots ensemble mean
        #
        # Convert to:
        #
        #     |RM - ensemble mean|
        # ----------------------------------------------------

        if (
            uncertainty_type
            == "distance"
        ):

            rm_values = (
                rm[
                    problem_idx,
                    :num_candidates,
                ]
            )

            values = np.abs(
                rm_values
                - values
            )

        rows.append(
            values
        )

    uncertainty = np.stack(
        rows
    )

    print(
        "NeuBoots uncertainty shape="
        f"{uncertainty.shape}"
    )

    print(
        "NeuBoots uncertainty "
        f"min={uncertainty.min():.6f} "
        f"mean={uncertainty.mean():.6f} "
        f"max={uncertainty.max():.6f}"
    )

    return uncertainty


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_ci(
        values,
        num_bootstrap=1000,
        seed=42,
):
    """
    Problem-level bootstrap confidence interval.
    """

    rng = (
        np.random.default_rng(
            seed
        )
    )

    n = len(
        values
    )

    indices = rng.integers(
        0,
        n,
        size=(
            num_bootstrap,
            n,
        ),
    )

    means = (
        values[
            indices
        ]
        .mean(
            axis=1
        )
    )

    (
        low,
        high,
    ) = np.quantile(
        means,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(
            low
        ),
        float(
            high
        ),
    )


# ============================================================
# Method score construction
# ============================================================

def build_method_scores(
        methods,
        rm,
        rnd,
        normalization_stats,
        neuboots_uncertainty=None,
        neuboots_normalization_key=None,
        caution_lambda=0.8,
        neuboots_lambda=0.8,
):
    """
    Independent calibration normalization.

    Reward:

        z_RM =
            (RM - mu_RM)
            / sigma_RM

    RND uncertainty:

        stored rnd_score = - uncertainty

        u_RND = -rnd_score

        z_RND =
            (u_RND - mu_RND)
            / sigma_RND

    NeuBoots:

        z_NB =
            (u_NB - mu_NB)
            / sigma_NB


    Method definitions
    ------------------

    RM:

        z_RM

    RND Pessimism:

        -z_RND

    RM + RND:

        z_RM
        - lambda_RND * z_RND

    NeuBoots Pessimism:

        -z_NB

    RM + NeuBoots:

        (1 - lambda_NB) * z_RM
        - lambda_NB * z_NB
    """

    scores = {}

    # ========================================================
    # Reward normalization
    # ========================================================

    if (
        "reward"
        not in normalization_stats
    ):
        raise ValueError(
            "Normalization stats do not "
            "contain 'reward'."
        )

    normalized_rm = (
        normalize_with_stats(
            rm,
            normalization_stats[
                "reward"
            ],
        )
    )

    # ========================================================
    # RND uncertainty normalization
    # ========================================================

    normalized_rnd_uncertainty = None

    need_rnd = any(
        method in methods
        for method in [
            "pessimism",
            "caution",
        ]
    )

    if need_rnd:

        if (
            "rnd_uncertainty"
            not in normalization_stats
        ):
            raise ValueError(
                "Normalization stats do not "
                "contain "
                "'rnd_uncertainty'."
            )

        # Stored:
        #
        # rnd_score = - uncertainty
        #
        # Convert back to positive uncertainty.

        rnd_uncertainty = (
            -rnd
        )

        normalized_rnd_uncertainty = (
            normalize_with_stats(
                rnd_uncertainty,
                normalization_stats[
                    "rnd_uncertainty"
                ],
            )
        )

    # ========================================================
    # NeuBoots uncertainty normalization
    # ========================================================

    normalized_nb_uncertainty = None

    need_neuboots = any(
        method in methods
        for method in [
            "neuboots_pessimism",
            "rm_neuboots",
        ]
    )

    if need_neuboots:

        if (
            neuboots_uncertainty
            is None
        ):
            raise ValueError(
                "NeuBoots uncertainty "
                "is required for selected "
                "NeuBoots methods."
            )

        if (
            neuboots_normalization_key
            is None
        ):
            raise ValueError(
                "NeuBoots normalization key "
                "is required."
            )

        if (
            neuboots_normalization_key
            not in normalization_stats
        ):
            raise ValueError(
                "Normalization stats do not "
                "contain "
                f"'{neuboots_normalization_key}'."
            )

        normalized_nb_uncertainty = (
            normalize_with_stats(
                neuboots_uncertainty,
                normalization_stats[
                    neuboots_normalization_key
                ],
            )
        )

    # ========================================================
    # Method scores
    # ========================================================

    for method in methods:

        # ----------------------------------------------------
        # Reward Model
        # ----------------------------------------------------

        if method == "rm":

            scores[
                method
            ] = normalized_rm

        # ----------------------------------------------------
        # RND pessimism only
        # ----------------------------------------------------

        elif method == "pessimism":

            scores[
                method
            ] = (
                -normalized_rnd_uncertainty
            )

        # ----------------------------------------------------
        # RM + Caution RND
        # ----------------------------------------------------

        elif method == "caution":

            scores[
                method
            ] = (
                normalized_rm
                - caution_lambda
                * normalized_rnd_uncertainty
            )

        # ----------------------------------------------------
        # NeuBoots uncertainty only
        # ----------------------------------------------------

        elif (
            method
            == "neuboots_pessimism"
        ):

            scores[
                method
            ] = (
                -normalized_nb_uncertainty
            )

        # ----------------------------------------------------
        # RM + NeuBoots
        #
        # Current experiment definition:
        #
        # (1-lambda) * RM
        # - lambda * uncertainty
        # ----------------------------------------------------

        elif (
            method
            == "rm_neuboots"
        ):

            scores[
                method
            ] = (
                (1.0 - neuboots_lambda)
                * normalized_rm
                - neuboots_lambda
                * normalized_nb_uncertainty
            )

        else:

            raise ValueError(
                f"Unsupported method: "
                f"{method}"
            )

    return scores


# ============================================================
# Evaluation
# ============================================================

def evaluate_methods(
        dataset_name,
        accuracy,
        method_scores,
        n_values,
        num_bootstrap=1000,
        bootstrap_seed=42,
):
    """
    Evaluate:

        N = 1, 2, 4, ..., 512
    """

    rows = []

    problem_idx = np.arange(
        accuracy.shape[0]
    )

    for (
        method,
        scores,
    ) in method_scores.items():

        for n in n_values:

            if (
                n
                > scores.shape[1]
            ):
                continue

            # First N candidates only.
            candidate_scores = (
                scores[
                    :,
                    :n,
                ]
            )

            selected_idx = (
                np.argmax(
                    candidate_scores,
                    axis=1,
                )
            )

            selected_accuracy = (
                accuracy[
                    problem_idx,
                    selected_idx,
                ]
            )

            mean_accuracy = float(
                selected_accuracy.mean()
            )

            (
                ci_low,
                ci_high,
            ) = bootstrap_ci(
                selected_accuracy,
                num_bootstrap=(
                    num_bootstrap
                ),
                seed=(
                    bootstrap_seed
                ),
            )

            rows.append(
                {
                    "dataset":
                        dataset_name,

                    "method":
                        method,

                    "method_display_name":
                        METHOD_DISPLAY_NAMES[
                            method
                        ],

                    "N":
                        int(
                            n
                        ),

                    "accuracy":
                        mean_accuracy,

                    "accuracy_percent":
                        mean_accuracy
                        * 100.0,

                    "ci_low":
                        ci_low
                        * 100.0,

                    "ci_high":
                        ci_high
                        * 100.0,
                }
            )

    return rows


# ============================================================
# Summary
# ============================================================

def build_summary(
        curve_df,
        caution_lambda,
        neuboots_lambda,
):
    """
    Peak is computed only over N_VALUES.
    """

    rows = []

    method_order = (
        curve_df[
            "method"
        ]
        .drop_duplicates()
        .tolist()
    )

    for method in method_order:

        subset = (
            curve_df[
                curve_df[
                    "method"
                ]
                == method
            ]
            .sort_values(
                "N"
            )
            .reset_index(
                drop=True
            )
        )

        peak_pos = int(
            subset[
                "accuracy"
            ]
            .to_numpy()
            .argmax()
        )

        peak_row = (
            subset.iloc[
                peak_pos
            ]
        )

        final_row = (
            subset.iloc[
                -1
            ]
        )

        degradation = (
            float(
                peak_row[
                    "accuracy"
                ]
            )
            - float(
                final_row[
                    "accuracy"
                ]
            )
        )

        if (
            method
            == "caution"
        ):
            method_weight = (
                caution_lambda
            )

        elif (
            method
            == "rm_neuboots"
        ):
            method_weight = (
                neuboots_lambda
            )

        else:
            method_weight = ""

        rows.append(
            {
                "dataset":
                    peak_row[
                        "dataset"
                    ],

                "method":
                    method,

                "method_display_name":
                    METHOD_DISPLAY_NAMES[
                        method
                    ],

                "method_weight":
                    method_weight,

                "peak_n":
                    int(
                        peak_row[
                            "N"
                        ]
                    ),

                "peak_accuracy":
                    float(
                        peak_row[
                            "accuracy"
                        ]
                    ),

                "peak_accuracy_percent":
                    float(
                        peak_row[
                            "accuracy_percent"
                        ]
                    ),

                "peak_ci_low":
                    float(
                        peak_row[
                            "ci_low"
                        ]
                    ),

                "peak_ci_high":
                    float(
                        peak_row[
                            "ci_high"
                        ]
                    ),

                "final_n":
                    int(
                        final_row[
                            "N"
                        ]
                    ),

                "final_accuracy":
                    float(
                        final_row[
                            "accuracy"
                        ]
                    ),

                "final_accuracy_percent":
                    float(
                        final_row[
                            "accuracy_percent"
                        ]
                    ),

                "final_ci_low":
                    float(
                        final_row[
                            "ci_low"
                        ]
                    ),

                "final_ci_high":
                    float(
                        final_row[
                            "ci_high"
                        ]
                    ),

                "degradation":
                    degradation,

                "degradation_percent":
                    degradation
                    * 100.0,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Console printing
# ============================================================

def print_normalization_stats(
        normalization_stats,
):

    print()

    print(
        "=" * 90
    )

    print(
        "Independent normalization statistics"
    )

    print(
        "=" * 90
    )

    for (
        name,
        stats,
    ) in normalization_stats.items():

        if not isinstance(
            stats,
            dict,
        ):
            continue

        if (
            "mean"
            not in stats
            or "std"
            not in stats
        ):
            continue

        n = stats.get(
            "num_values",
            "N/A",
        )

        print(
            f"{name:<45}"
            f"mean="
            f"{stats['mean']:>12.6f}  "
            f"std="
            f"{stats['std']:>12.6f}  "
            f"n={n}"
        )

    print(
        "=" * 90
    )


def print_summary(
        summary_df,
):

    print()

    print(
        "=" * 96
    )

    print(
        "Figure 3 evaluation summary"
    )

    print(
        "=" * 96
    )

    print(
        f"{'Method':<30}"
        f"{'Peak':>10}"
        f"{'Peak N':>10}"
        f"{'Final':>10}"
        f"{'Final N':>10}"
        f"{'Deg.':>10}"
    )

    print(
        "-" * 96
    )

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['method_display_name']:<30}"
            f"{row['peak_accuracy_percent']:>9.1f}%"
            f"{int(row['peak_n']):>10d}"
            f"{row['final_accuracy_percent']:>9.1f}%"
            f"{int(row['final_n']):>10d}"
            f"{row['degradation_percent']:>9.1f}"
        )

    print(
        "=" * 96
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # ========================================================
    # Dataset inputs
    # ========================================================

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--caution-detailed",
        required=True,
        type=Path,
        help=(
            "Original Caution "
            "detailed_candidates.json"
        ),
    )

    parser.add_argument(
        "--normalization-stats",
        required=True,
        type=Path,
        help=(
            "Normalization statistics fitted "
            "on an independent response set."
        ),
    )

    parser.add_argument(
        "--neuboots-results",
        type=Path,
        default=None,
        help=(
            "NeuBoots scored "
            "candidate JSONL."
        ),
    )

    # ========================================================
    # Methods
    # ========================================================

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            SUPPORTED_METHODS
        ),
        default=[
            "rm",
            "pessimism",
            "caution",
        ],
    )

    # ========================================================
    # Lambdas
    # ========================================================

    parser.add_argument(
        "--caution-lambda",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--neuboots-lambda",
        type=float,
        default=0.8,
    )

    # ========================================================
    # NeuBoots uncertainty selection
    # ========================================================

    parser.add_argument(
        "--neuboots-uncertainty-type",
        type=str,
        choices=(
            NEUBOOTS_UNCERTAINTY_TYPES
        ),
        default="std",
        help=(
            "NeuBoots uncertainty definition."
        ),
    )

    parser.add_argument(
        "--neuboots-alpha",
        type=float,
        choices=(
            NEUBOOTS_ALPHAS
        ),
        default=None,
        help=(
            "Alpha for quantile-based "
            "NeuBoots uncertainty."
        ),
    )

    # ========================================================
    # Bootstrap
    # ========================================================

    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # NeuBoots configuration validation
    # ========================================================

    quantile_types = {
        "median_quantile",
        "avg_quantile",
        "avg_quantile_avg",
    }

    if (
        args.neuboots_uncertainty_type
        in quantile_types
        and args.neuboots_alpha
        is None
    ):
        raise ValueError(
            "--neuboots-alpha is required "
            "when using a quantile-based "
            "NeuBoots uncertainty type."
        )

    if (
        args.neuboots_uncertainty_type
        not in quantile_types
        and args.neuboots_alpha
        is not None
    ):
        raise ValueError(
            "--neuboots-alpha should only "
            "be supplied for "
            "median_quantile, "
            "avg_quantile, or "
            "avg_quantile_avg."
        )

    # ========================================================
    # Check whether NeuBoots data is required
    # ========================================================

    use_neuboots = any(
        method in args.methods
        for method in [
            "neuboots_pessimism",
            "rm_neuboots",
        ]
    )

    if (
        use_neuboots
        and args.neuboots_results
        is None
    ):
        raise ValueError(
            "--neuboots-results is required "
            "when using a NeuBoots method."
        )

    # ========================================================
    # Output directory
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load independent normalization stats
    # ========================================================

    normalization_stats = (
        load_normalization_stats(
            args.normalization_stats
        )
    )

    print_normalization_stats(
        normalization_stats
    )

    # ========================================================
    # Load Caution candidate data
    # ========================================================

    (
        problem_ids,
        accuracy,
        rm,
        rnd,
    ) = load_caution_dataset(
        args.caution_detailed
    )

    # ========================================================
    # NeuBoots normalization key
    # ========================================================

    neuboots_normalization_key = None

    if use_neuboots:

        neuboots_normalization_key = (
            get_neuboots_normalization_key(
                uncertainty_type=(
                    args.neuboots_uncertainty_type
                ),
                alpha=(
                    args.neuboots_alpha
                ),
            )
        )

        print()

        print(
            "Selected NeuBoots uncertainty:"
        )

        print(
            "  type = "
            f"{args.neuboots_uncertainty_type}"
        )

        if (
            args.neuboots_alpha
            is not None
        ):
            print(
                "  alpha = "
                f"{args.neuboots_alpha:.1f}"
            )

        print(
            "  normalization key = "
            f"{neuboots_normalization_key}"
        )

    # ========================================================
    # Load NeuBoots uncertainty
    # ========================================================

    neuboots_uncertainty = None

    if use_neuboots:

        neuboots_uncertainty = (
            load_neuboots_results(
                path=(
                    args.neuboots_results
                ),
                problem_ids=(
                    problem_ids
                ),
                num_candidates=(
                    accuracy.shape[1]
                ),
                uncertainty_type=(
                    args.neuboots_uncertainty_type
                ),
                rm=rm,
                alpha=(
                    args.neuboots_alpha
                ),
            )
        )

    # ========================================================
    # Build normalized selection scores
    # ========================================================

    method_scores = (
        build_method_scores(
            methods=(
                args.methods
            ),
            rm=rm,
            rnd=rnd,
            normalization_stats=(
                normalization_stats
            ),
            neuboots_uncertainty=(
                neuboots_uncertainty
            ),
            neuboots_normalization_key=(
                neuboots_normalization_key
            ),
            caution_lambda=(
                args.caution_lambda
            ),
            neuboots_lambda=(
                args.neuboots_lambda
            ),
        )
    )

    # ========================================================
    # Valid N values
    # ========================================================

    valid_n_values = [
        n
        for n in N_VALUES
        if (
            n
            <= accuracy.shape[1]
        )
    ]

    # ========================================================
    # Evaluate
    # ========================================================

    rows = evaluate_methods(
        dataset_name=(
            args.dataset
        ),
        accuracy=(
            accuracy
        ),
        method_scores=(
            method_scores
        ),
        n_values=(
            valid_n_values
        ),
        num_bootstrap=(
            args.num_bootstrap
        ),
        bootstrap_seed=(
            args.bootstrap_seed
        ),
    )

    curve_df = pd.DataFrame(
        rows
    )

    summary_df = (
        build_summary(
            curve_df=(
                curve_df
            ),
            caution_lambda=(
                args.caution_lambda
            ),
            neuboots_lambda=(
                args.neuboots_lambda
            ),
        )
    )

    # ========================================================
    # Save curve CSV
    # ========================================================

    curve_path = (
        args.output_dir
        / "curves.csv"
    )

    curve_df.to_csv(
        curve_path,
        index=False,
    )

    # ========================================================
    # Save summary CSV
    # ========================================================

    summary_path = (
        args.output_dir
        / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # NeuBoots metadata
    # ========================================================

    neuboots_definition = None

    if use_neuboots:

        neuboots_definition = (
            get_neuboots_uncertainty_definition(
                uncertainty_type=(
                    args.neuboots_uncertainty_type
                ),
                alpha=(
                    args.neuboots_alpha
                ),
            )
        )

    # ========================================================
    # Save full result metadata
    # ========================================================

    result = {
        "dataset":
            args.dataset,

        "methods":
            args.methods,

        "method_display_names": {
            method:
                METHOD_DISPLAY_NAMES[
                    method
                ]
            for method
            in args.methods
        },

        "num_problems":
            int(
                accuracy.shape[0]
            ),

        "num_candidates":
            int(
                accuracy.shape[1]
            ),

        "n_values":
            valid_n_values,

        "caution_lambda":
            args.caution_lambda,

        "neuboots_lambda":
            args.neuboots_lambda,

        # ----------------------------------------------------
        # NeuBoots uncertainty configuration
        # ----------------------------------------------------

        "neuboots_uncertainty_type":
            (
                args.neuboots_uncertainty_type
                if use_neuboots
                else None
            ),

        "neuboots_alpha":
            (
                args.neuboots_alpha
                if use_neuboots
                else None
            ),

        "neuboots_uncertainty_definition":
            neuboots_definition,

        "neuboots_normalization_key":
            neuboots_normalization_key,

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        "normalization_scope":
            "independent_response_set",

        "normalization_stats_path":
            str(
                args.normalization_stats
            ),

        "normalization_stats":
            normalization_stats,

        # ----------------------------------------------------
        # Score definitions
        # ----------------------------------------------------

        "score_definitions": {
            "rm":
                "normalized_reward",

            "pessimism":
                (
                    "-normalized_"
                    "rnd_uncertainty"
                ),

            "caution":
                (
                    "normalized_reward "
                    "- caution_lambda * "
                    "normalized_rnd_uncertainty"
                ),

            "neuboots_pessimism":
                (
                    "-normalized_"
                    "neuboots_uncertainty"
                ),

            "rm_neuboots":
                (
                    "(1 - neuboots_lambda) * "
                    "normalized_reward "
                    "- neuboots_lambda * "
                    "normalized_neuboots_uncertainty"
                ),
        },

        # ----------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------

        "num_bootstrap":
            args.num_bootstrap,

        "bootstrap_seed":
            args.bootstrap_seed,

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        "summary":
            json.loads(
                summary_df.to_json(
                    orient="records"
                )
            ),

        "curves":
            json.loads(
                curve_df.to_json(
                    orient="records"
                )
            ),
    }

    json_path = (
        args.output_dir
        / "results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # ========================================================
    # Print
    # ========================================================

    print_summary(
        summary_df
    )

    print()

    if use_neuboots:

        print(
            "NeuBoots configuration"
        )

        print(
            "  uncertainty type : "
            f"{args.neuboots_uncertainty_type}"
        )

        if (
            args.neuboots_alpha
            is not None
        ):
            print(
                "  alpha            : "
                f"{args.neuboots_alpha:.1f}"
            )

        print(
            "  normalization key : "
            f"{neuboots_normalization_key}"
        )

        print(
            "  lambda            : "
            f"{args.neuboots_lambda}"
        )

        print()

    print(
        f"Saved curves : "
        f"{curve_path}"
    )

    print(
        f"Saved summary: "
        f"{summary_path}"
    )

    print(
        f"Saved JSON   : "
        f"{json_path}"
    )


if __name__ == "__main__":
    main()