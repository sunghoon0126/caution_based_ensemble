#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import ijson
import numpy as np
import pandas as pd


# ============================================================
# Exact N grid used in the previous Figure 3 reproduction
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
    "pessimism": "Pessimism",
    "caution": "RM + Pessimism",
    "neuboots_pessimism": "NeuBoots Pessimism",
    "rm_neuboots": "RM + NeuBoots",
}


# ============================================================
# Caution detailed candidate loader
# ============================================================

def candidate_key(value):
    """
    Preserve the candidate ordering used by the
    previous Figure 3 reproduction code.
    """

    text = str(value)
    numbers = re.findall(r"\d+", text)

    if numbers:
        return int(numbers[-1]), text

    return 10**12, text


def iter_top_level(path):
    """
    Stream either a top-level dictionary or list
    without loading the entire detailed_candidates.json
    into memory.
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
    Load exactly the same three candidate-level
    quantities used by the previous Figure 3 code:

        accuracy
        reward_model_score
        rnd_score

    Returned arrays have shape:

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
            f"Truncating all problems to "
            f"{min_candidates} candidates."
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

    Expected JSONL fields:

        instance_id
        all_neuboots_uncertainty

    Candidate ordering must be the same as
    the original Caution generation.
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

        print(
            "\nCaution / NeuBoots IDs "
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
# Normalization
# ============================================================

def global_zscore(
        values,
        name,
):
    """
    IMPORTANT:

    This exactly follows the previous
    Figure 3 reproduction:

        values.mean()
        values.std()

    on the full [problem, candidate] matrix.

    Therefore normalization is GLOBAL over
    all problem-candidate pairs in a dataset,
    not per-problem.
    """

    mean = float(
        values.mean()
    )

    std = float(
        values.std()
    )

    if std == 0.0:
        raise ValueError(
            f"{name} has zero standard deviation."
        )

    normalized = (
        values - mean
    ) / std

    return (
        normalized,
        mean,
        std,
    )


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_ci(
        values,
        num_bootstrap=1000,
        seed=42,
):
    """
    Same bootstrap implementation used by
    the previous Figure 3 reproduction.
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
# Score construction
# ============================================================

def build_method_scores(
        methods,
        rm,
        rnd,
        neuboots_uncertainty=None,
        caution_lambda=0.8,
        neuboots_lambda=0.8,
):
    """
    Previous Figure 3 definitions:

        RM:
            z(RM)

        Pessimism:
            z(RND)

        RM + Pessimism:
            (1-lambda) * z(RM)
            + lambda * z(RND)

    NeuBoots extension:

        NB pessimism score:
            - uncertainty

        NeuBoots Pessimism:
            z(- uncertainty)

        RM + NeuBoots:
            (1-lambda_NB) * z(RM)
            + lambda_NB * z(- uncertainty)
    """

    scores = {}

    normalization = {}

    need_rm = any(
        method in methods
        for method in [
            "rm",
            "caution",
            "rm_neuboots",
        ]
    )

    need_rnd = any(
        method in methods
        for method in [
            "pessimism",
            "caution",
        ]
    )

    need_neuboots = any(
        method in methods
        for method in [
            "neuboots_pessimism",
            "rm_neuboots",
        ]
    )

    z_rm = None
    z_rnd = None
    z_neuboots = None

    if need_rm:

        (
            z_rm,
            rm_mean,
            rm_std,
        ) = global_zscore(
            rm,
            "RM score",
        )

        normalization["rm"] = {
            "mean": rm_mean,
            "std": rm_std,
        }

    if need_rnd:

        (
            z_rnd,
            rnd_mean,
            rnd_std,
        ) = global_zscore(
            rnd,
            "RND score",
        )

        normalization["rnd"] = {
            "mean": rnd_mean,
            "std": rnd_std,
        }

    if need_neuboots:

        if neuboots_uncertainty is None:
            raise ValueError(
                "NeuBoots uncertainty is required "
                "for selected NeuBoots methods."
            )

        # Lower uncertainty is better.
        # Convert it into a score for argmax.
        neuboots_pessimism = (
            -neuboots_uncertainty
        )

        (
            z_neuboots,
            nb_mean,
            nb_std,
        ) = global_zscore(
            neuboots_pessimism,
            "NeuBoots pessimism score",
        )

        normalization[
            "neuboots_pessimism"
        ] = {
            "mean": nb_mean,
            "std": nb_std,
        }

    for method in methods:

        if method == "rm":
            scores[method] = z_rm

        elif method == "pessimism":
            scores[method] = z_rnd

        elif method == "caution":
            scores[method] = (
                (
                    1.0
                    - caution_lambda
                )
                * z_rm
                + caution_lambda
                * z_rnd
            )

        elif (
            method
            == "neuboots_pessimism"
        ):
            scores[method] = (
                z_neuboots
            )

        elif method == "rm_neuboots":
            scores[method] = (
                (
                    1.0
                    - neuboots_lambda
                )
                * z_rm
                + neuboots_lambda
                * z_neuboots
            )

        else:
            raise ValueError(
                f"Unsupported method: {method}"
            )

    return (
        scores,
        normalization,
    )


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
    Evaluate only the specified N grid.

    This intentionally does NOT evaluate
    every N from 1 to 512.
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


def build_summary(
        curve_df,
        caution_lambda,
        neuboots_lambda,
):
    """
    Peak is computed ONLY over the evaluated
    N grid, exactly like the previous
    reproduction workflow.
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
                curve_df["method"]
                == method
            ]
            .sort_values("N")
            .reset_index(drop=True)
        )

        peak_idx = int(
            subset[
                "accuracy"
            ].idxmax()
        )

        peak_row = (
            subset.loc[
                peak_idx
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

        elif method == "rm_neuboots":
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
# Printing
# ============================================================

def print_summary(
        summary_df,
):
    print()
    print("=" * 96)

    print(
        "Figure 3 reproduction summary"
    )

    print("=" * 96)

    print(
        f"{'Method':<28}"
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
            f"{row['method_display_name']:<28}"
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
        "--neuboots-results",
        type=Path,
        default=None,
        help=(
            "NeuBoots scored_candidates.jsonl"
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        problem_ids,
        accuracy,
        rm,
        rnd,
    ) = load_caution_dataset(
        args.caution_detailed
    )

    neuboots_uncertainty = None

    if use_neuboots:

        neuboots_uncertainty = (
            load_neuboots_results(
                path=(
                    args.neuboots_results
                ),
                problem_ids=problem_ids,
                num_candidates=(
                    accuracy.shape[1]
                ),
            )
        )

    (
        method_scores,
        normalization,
    ) = build_method_scores(
        methods=args.methods,
        rm=rm,
        rnd=rnd,
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

    valid_n_values = [
        n
        for n in N_VALUES
        if n <= accuracy.shape[1]
    ]

    rows = evaluate_methods(
        dataset_name=args.dataset,
        accuracy=accuracy,
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

    # ----------------------------------------
    # Save curve CSV
    # ----------------------------------------

    curve_path = (
        args.output_dir
        / "curves.csv"
    )

    curve_df.to_csv(
        curve_path,
        index=False,
    )

    # ----------------------------------------
    # Save summary CSV
    # ----------------------------------------

    summary_path = (
        args.output_dir
        / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ----------------------------------------
    # Save metadata / full result JSON
    # ----------------------------------------

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
            "global_dataset_problem_candidate",

        "normalization":
            normalization,

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