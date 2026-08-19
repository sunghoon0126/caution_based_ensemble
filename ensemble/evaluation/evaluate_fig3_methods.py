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


SUPPORTED_METHODS = [
    "rm",
    "pessimism",
    "caution",
    "neuboots_pessimism",
    "rm_neuboots",
]


METHOD_DISPLAY_NAMES = {
    "rm": "Reward Model",
    "pessimism": "RND Pessimism",
    "caution": "RM + RND Pessimism",
    "neuboots_pessimism": "NeuBoots Pessimism",
    "rm_neuboots": "RM + NeuBoots",
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
    numbers = re.findall(r"\d+", text)

    if numbers:
        return int(numbers[-1]), text

    return 10**12, text


def iter_top_level(path):
    """
    Stream a top-level JSON dictionary or list.
    """

    with open(path, "rb") as f:
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
                yield str(i), item

        else:
            raise ValueError(
                f"Unsupported JSON root: {path}"
            )


def to_accuracy(value):
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return float(value)

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if isinstance(value, str):
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

        return float(value)

    raise TypeError(
        f"Unsupported accuracy value: {value}"
    )


def load_caution_dataset(path):
    """
    Load candidate-level:

        accuracy
        reward_model_score
        rnd_score

    rnd_score in the stored Caution artifacts is
    treated as a pessimism score:

        rnd_score = - uncertainty

    Arrays:
        [num_problems, num_candidates]
    """

    problem_ids = []

    accuracy_rows = []
    rm_rows = []
    rnd_rows = []

    candidate_counts = []

    print(
        f"Reading Caution candidates: {path}"
    )

    for problem_idx, (
        problem_id,
        problem,
    ) in enumerate(
        iter_top_level(path),
        start=1,
    ):
        responses = problem["responses"]

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
                    f"field in problem {problem_idx}"
                )

            rm_values.append(
                float(rm)
            )

            rnd_values.append(
                float(rnd)
            )

            accuracy_values.append(
                to_accuracy(
                    accuracy
                )
            )

        problem_ids.append(
            str(problem_id)
        )

        candidate_counts.append(
            len(candidates)
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

        if problem_idx % 100 == 0:
            print(
                f"  loaded {problem_idx} problems"
            )

    if not candidate_counts:
        raise RuntimeError(
            "No problems were loaded."
        )

    min_candidates = min(
        candidate_counts
    )

    if len(
        set(candidate_counts)
    ) != 1:
        print(
            "Warning: candidate counts differ. "
            f"Truncating to {min_candidates}."
        )

    accuracy = np.stack(
        [
            row[:min_candidates]
            for row in accuracy_rows
        ]
    )

    rm = np.stack(
        [
            row[:min_candidates]
            for row in rm_rows
        ]
    )

    rnd = np.stack(
        [
            row[:min_candidates]
            for row in rnd_rows
        ]
    )

    print(
        f"Problems={accuracy.shape[0]}, "
        f"Candidates={accuracy.shape[1]}"
    )

    return (
        problem_ids,
        accuracy,
        rm,
        rnd,
    )


# ============================================================
# NeuBoots result loader
# ============================================================

def load_neuboots_results(
        path,
        problem_ids,
        num_candidates,
):
    """
    Load NeuBoots candidate uncertainty.

    Expected JSONL:

        instance_id
        all_neuboots_uncertainty

    Candidate ordering must match the
    Caution candidate ordering.
    """

    data = {}

    print(
        f"Reading NeuBoots scores: {path}"
    )

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

            instance_id = str(
                item["instance_id"]
            )

            if instance_id in data:
                raise ValueError(
                    "Duplicate NeuBoots "
                    f"instance_id: {instance_id}"
                )

            data[instance_id] = np.asarray(
                item[
                    "all_neuboots_uncertainty"
                ],
                dtype=np.float64,
            )

    caution_ids = set(
        problem_ids
    )

    neuboots_ids = set(
        data
    )

    if caution_ids != neuboots_ids:

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
            f"Missing NeuBoots IDs: "
            f"{len(missing)}"
        )

        print(
            f"Extra NeuBoots IDs: "
            f"{len(extra)}"
        )

        if missing:
            print(
                "Example missing IDs:",
                list(missing)[:5],
            )

        if extra:
            print(
                "Example extra IDs:",
                list(extra)[:5],
            )

        raise ValueError(
            "Cannot safely align "
            "NeuBoots candidates."
        )

    rows = []

    for problem_id in problem_ids:

        uncertainty = data[
            problem_id
        ]

        if len(
            uncertainty
        ) < num_candidates:
            raise ValueError(
                f"{problem_id}: "
                "NeuBoots has only "
                f"{len(uncertainty)} candidates, "
                f"expected at least "
                f"{num_candidates}."
            )

        rows.append(
            uncertainty[
                :num_candidates
            ]
        )

    uncertainty = np.stack(
        rows
    )

    print(
        "NeuBoots uncertainty shape="
        f"{uncertainty.shape}"
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

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    indices = rng.integers(
        0,
        n,
        size=(
            num_bootstrap,
            n,
        ),
    )

    means = (
        values[indices]
        .mean(axis=1)
    )

    low, high = np.quantile(
        means,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
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
        caution_lambda=0.8,
        neuboots_lambda=0.8,
):
    """
    Paper-style normalization.

    IMPORTANT
    ---------
    Mean/std are NOT estimated from the current
    evaluation dataset.

    They must already have been fitted on an
    independent response set.

    Reward:
        r_norm =
            (r - mu_r_cal)
            / sigma_r_cal

    RND:
        stored rnd_score = - uncertainty

        u_rnd = -rnd_score

        u_rnd_norm =
            (u_rnd - mu_rnd_cal)
            / sigma_rnd_cal

    NeuBoots:
        u_nb_norm =
            (u_nb - mu_nb_cal)
            / sigma_nb_cal


    Method definitions
    ------------------

    RM:
        r_norm

    RND Pessimism:
        -u_rnd_norm

    RM + RND:
        r_norm
        - lambda_RND * u_rnd_norm

    NeuBoots Pessimism:
        -u_nb_norm

    RM + NeuBoots:
        r_norm
        - lambda_NB * u_nb_norm
    """

    scores = {}

    # --------------------------------------------------------
    # Reward normalization
    # --------------------------------------------------------

    if "reward" not in normalization_stats:
        raise ValueError(
            "Normalization stats do not "
            "contain 'reward'."
        )

    normalized_rm = normalize_with_stats(
        rm,
        normalization_stats[
            "reward"
        ],
    )

    # --------------------------------------------------------
    # RND uncertainty normalization
    # --------------------------------------------------------

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
                "contain 'rnd_uncertainty'."
            )

        # Stored Caution convention:
        #
        # rnd_score = - uncertainty
        #
        # Convert it back into positive uncertainty.
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

    # --------------------------------------------------------
    # NeuBoots uncertainty normalization
    # --------------------------------------------------------

    normalized_nb_uncertainty = None

    need_neuboots = any(
        method in methods
        for method in [
            "neuboots_pessimism",
            "rm_neuboots",
        ]
    )

    if need_neuboots:

        if neuboots_uncertainty is None:
            raise ValueError(
                "NeuBoots uncertainty is required "
                "for selected NeuBoots methods."
            )

        if (
            "neuboots_uncertainty"
            not in normalization_stats
        ):
            raise ValueError(
                "Normalization stats do not "
                "contain "
                "'neuboots_uncertainty'."
            )

        normalized_nb_uncertainty = (
            normalize_with_stats(
                neuboots_uncertainty,
                normalization_stats[
                    "neuboots_uncertainty"
                ],
            )
        )

    # --------------------------------------------------------
    # Method scores
    # --------------------------------------------------------

    for method in methods:

        if method == "rm":

            scores[method] = (
                normalized_rm
            )

        elif method == "pessimism":

            scores[method] = (
                -normalized_rnd_uncertainty
            )

        elif method == "caution":

            scores[method] = (
                normalized_rm
                - caution_lambda
                * normalized_rnd_uncertainty
            )

        elif (
            method
            == "neuboots_pessimism"
        ):

            scores[method] = (
                -normalized_nb_uncertainty
            )

        elif method == "rm_neuboots":

            scores[method] = (
                normalized_rm
                - neuboots_lambda
                * normalized_nb_uncertainty
            )

        else:
            raise ValueError(
                f"Unsupported method: {method}"
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
    Evaluate only:

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

            if n > scores.shape[1]:
                continue

            candidate_scores = (
                scores[:, :n]
            )

            selected_idx = np.argmax(
                candidate_scores,
                axis=1,
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
                        int(n),

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
                ] == method
            ]
            .sort_values("N")
            .reset_index(
                drop=True
            )
        )

        peak_pos = int(
            subset[
                "accuracy"
            ].to_numpy().argmax()
        )

        peak_row = (
            subset.iloc[
                peak_pos
            ]
        )

        final_row = (
            subset.iloc[-1]
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

        if method == "caution":
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
                        peak_row["N"]
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
                        final_row["N"]
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
    print("=" * 72)
    print(
        "Independent normalization statistics"
    )
    print("=" * 72)

    for name, stats in (
        normalization_stats.items()
    ):
        if not isinstance(
            stats,
            dict,
        ):
            continue

        if (
            "mean" not in stats
            or "std" not in stats
        ):
            continue

        n = stats.get(
            "num_values",
            "N/A",
        )

        print(
            f"{name:<24}"
            f"mean={stats['mean']:>12.6f}  "
            f"std={stats['std']:>12.6f}  "
            f"n={n}"
        )

    print("=" * 72)


def print_summary(
        summary_df,
):
    print()
    print("=" * 96)

    print(
        "Figure 3 evaluation summary"
    )

    print("=" * 96)

    print(
        f"{'Method':<30}"
        f"{'Peak':>10}"
        f"{'Peak N':>10}"
        f"{'Final':>10}"
        f"{'Final N':>10}"
        f"{'Deg.':>10}"
    )

    print("-" * 96)

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

    print("=" * 96)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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
            "NeuBoots "
            "scored_candidates.jsonl"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=[
            "rm",
            "pessimism",
            "caution",
        ],
    )

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

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate NeuBoots requirement
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load independent normalization stats
    # --------------------------------------------------------

    normalization_stats = (
        load_normalization_stats(
            args.normalization_stats
        )
    )

    print_normalization_stats(
        normalization_stats
    )

    # --------------------------------------------------------
    # Load Caution candidate data
    # --------------------------------------------------------

    (
        problem_ids,
        accuracy,
        rm,
        rnd,
    ) = load_caution_dataset(
        args.caution_detailed
    )

    # --------------------------------------------------------
    # Load NeuBoots uncertainty if needed
    # --------------------------------------------------------

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
            )
        )

    # --------------------------------------------------------
    # Build normalized selection scores
    # --------------------------------------------------------

    method_scores = (
        build_method_scores(
            methods=args.methods,
            rm=rm,
            rnd=rnd,
            normalization_stats=(
                normalization_stats
            ),
            neuboots_uncertainty=(
                neuboots_uncertainty
            ),
            caution_lambda=(
                args.caution_lambda
            ),
            neuboots_lambda=(
                args.neuboots_lambda
            ),
        )
    )

    # --------------------------------------------------------
    # Valid N values
    # --------------------------------------------------------

    valid_n_values = [
        n
        for n in N_VALUES
        if n <= accuracy.shape[1]
    ]

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

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

    summary_df = build_summary(
        curve_df=curve_df,
        caution_lambda=(
            args.caution_lambda
        ),
        neuboots_lambda=(
            args.neuboots_lambda
        ),
    )

    # --------------------------------------------------------
    # Save curve CSV
    # --------------------------------------------------------

    curve_path = (
        args.output_dir
        / "curves.csv"
    )

    curve_df.to_csv(
        curve_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_path = (
        args.output_dir
        / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save full result metadata
    # --------------------------------------------------------

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
            for method in args.methods
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

        "normalization_scope":
            "independent_response_set",

        "normalization_stats_path":
            str(
                args.normalization_stats
            ),

        "normalization_stats":
            normalization_stats,

        "score_definitions": {
            "rm":
                "normalized_reward",

            "pessimism":
                "-normalized_rnd_uncertainty",

            "caution":
                (
                    "normalized_reward "
                    "- caution_lambda * "
                    "normalized_rnd_uncertainty"
                ),

            "neuboots_pessimism":
                "-normalized_neuboots_uncertainty",

            "rm_neuboots":
                (
                    "normalized_reward "
                    "- neuboots_lambda * "
                    "normalized_neuboots_uncertainty"
                ),
        },

        "num_bootstrap":
            args.num_bootstrap,

        "bootstrap_seed":
            args.bootstrap_seed,

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

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        summary_df
    )

    print()
    print(
        f"Saved curves : {curve_path}"
    )

    print(
        f"Saved summary: {summary_path}"
    )

    print(
        f"Saved JSON   : {json_path}"
    )


if __name__ == "__main__":
    main()