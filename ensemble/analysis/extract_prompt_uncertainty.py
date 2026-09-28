#!/usr/bin/env python3

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# Utility
# ============================================================

def candidate_digest(samples):
    """
    Compute one digest for the ordered candidate list.

    This lets us verify that:
        input_jsonl candidate 1..512
    and
        Caution detailed response 1..512

    have exactly the same ordering without keeping all
    response strings in memory.
    """

    h = hashlib.sha256()

    for sample in samples:
        if not isinstance(sample, str):
            sample = str(sample)

        h.update(
            sample.encode(
                "utf-8",
                errors="replace",
            )
        )

        # Candidate separator
        h.update(b"\0")

    return h.hexdigest()


def to_accuracy(value):
    """
    Convert the various accuracy representations
    used by the evaluation artifacts into float.
    """

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip().lower()

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


def iter_detailed_json(path):
    """
    Stream the top-level Caution detailed JSON.

    Expected structure:

        {
            problem_uuid: {
                ...
                "responses": {...}
            },
            ...
        }

    We intentionally stream it because detailed_candidates
    can be very large.
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

            for idx, item in enumerate(
                ijson.items(
                    f,
                    "item",
                )
            ):
                yield str(idx), item

        else:
            raise ValueError(
                f"Unsupported detailed JSON root: {path}"
            )


# ============================================================
# Load prompt / candidate ordering
# ============================================================

def load_input_metadata(
        path,
        tokenizer,
):
    """
    Load:

        problem_id
        prompt
        prompt character length
        prompt token length
        candidate count
        candidate ordering digest

    from the NeuBoots scorer input JSONL.
    """

    metadata = {}

    print()
    print(
        f"Loading input JSONL: {path}"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_idx, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            problem_id = str(
                item["instance_id"]
            )

            if problem_id in metadata:
                raise ValueError(
                    f"Duplicate instance_id: "
                    f"{problem_id}"
                )

            prompt = item["prompt"]

            samples = item[
                "all_samples"
            ]

            # --------------------------------------------
            # Prompt character length
            # --------------------------------------------

            prompt_char_length = len(
                prompt
            )

            # --------------------------------------------
            # Prompt token length
            #
            # Do NOT truncate here.
            #
            # We want the actual prompt length rather
            # than the model's truncated input length.
            # --------------------------------------------

            tokenized = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )

            prompt_token_length = len(
                tokenized[
                    "input_ids"
                ]
            )

            metadata[
                problem_id
            ] = {
                "prompt":
                    prompt,

                "prompt_char_length":
                    prompt_char_length,

                "prompt_token_length":
                    prompt_token_length,

                "num_candidates":
                    len(samples),

                "candidate_digest":
                    candidate_digest(
                        samples
                    ),
            }

            if line_idx % 100 == 0:
                print(
                    f"  loaded "
                    f"{line_idx} problems"
                )

    if not metadata:
        raise RuntimeError(
            "No input problems were loaded."
        )

    print(
        f"Input problems: "
        f"{len(metadata)}"
    )

    return metadata


# ============================================================
# Load NeuBoots results
# ============================================================

def load_neuboots_results(path):
    """
    Expected JSONL fields:

        instance_id
        all_neuboots_mc_mean
        all_neuboots_uncertainty
    """

    results = {}

    print()
    print(
        f"Loading NeuBoots results: {path}"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_idx, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            problem_id = str(
                item["instance_id"]
            )

            if problem_id in results:
                raise ValueError(
                    f"Duplicate NeuBoots "
                    f"instance_id: {problem_id}"
                )

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
                    f"{problem_id}: "
                    "NeuBoots mean / uncertainty "
                    "length mismatch."
                )

            results[
                problem_id
            ] = {
                "mean":
                    mc_mean,

                "uncertainty":
                    uncertainty,

                "num_mc":
                    item.get(
                        "num_mc"
                    ),
            }

            if line_idx % 100 == 0:
                print(
                    f"  loaded "
                    f"{line_idx} problems"
                )

    if not results:
        raise RuntimeError(
            "No NeuBoots results were loaded."
        )

    print(
        f"NeuBoots problems: "
        f"{len(results)}"
    )

    return results


# ============================================================
# Main extraction
# ============================================================

def extract(
        dataset_name,
        trial,
        input_jsonl,
        caution_detailed,
        neuboots_results,
        tokenizer_name,
        output_dir,
):

    print("=" * 80)
    print(
        "Prompt length / uncertainty extraction"
    )
    print("=" * 80)

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Trial: {trial}"
    )

    # ========================================================
    # 1. Tokenizer
    # ========================================================

    print()
    print(
        f"Loading tokenizer: "
        f"{tokenizer_name}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            tokenizer_name,
            use_fast=True,
            trust_remote_code=True,
        )
    )

    # ========================================================
    # 2. Input metadata
    # ========================================================

    input_metadata = (
        load_input_metadata(
            path=input_jsonl,
            tokenizer=tokenizer,
        )
    )

    # ========================================================
    # 3. NeuBoots results
    # ========================================================

    nb_results = (
        load_neuboots_results(
            neuboots_results
        )
    )

    # ========================================================
    # 4. Validate problem IDs
    # ========================================================

    input_ids = set(
        input_metadata
    )

    nb_ids = set(
        nb_results
    )

    if input_ids != nb_ids:

        missing_nb = (
            input_ids
            - nb_ids
        )

        extra_nb = (
            nb_ids
            - input_ids
        )

        print()
        print(
            f"Missing NeuBoots IDs: "
            f"{len(missing_nb)}"
        )

        print(
            f"Extra NeuBoots IDs: "
            f"{len(extra_nb)}"
        )

        if missing_nb:
            print(
                "Example missing:",
                list(
                    missing_nb
                )[:5],
            )

        if extra_nb:
            print(
                "Example extra:",
                list(
                    extra_nb
                )[:5],
            )

        raise RuntimeError(
            "Input / NeuBoots problem IDs "
            "do not match."
        )

    # ========================================================
    # 5. Output setup
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_output = (
        output_dir
        / (
            f"{dataset_name}_"
            f"trial{trial}_"
            f"candidate_level.csv.gz"
        )
    )

    problem_output = (
        output_dir
        / (
            f"{dataset_name}_"
            f"trial{trial}_"
            f"problem_level.csv"
        )
    )

    candidate_fields = [
        "dataset",
        "trial",
        "problem_id",
        "candidate_idx",
        "prompt_char_length",
        "prompt_token_length",
        "reward_model_score",
        "accuracy",
        "caution_rnd_score",
        "caution_uncertainty",
        "neuboots_mean",
        "neuboots_uncertainty",
    ]

    problem_rows = []

    detailed_ids = set()

    num_candidate_rows = 0

    # ========================================================
    # 6. Stream Caution detailed candidates
    # ========================================================

    print()
    print(
        f"Streaming Caution detailed: "
        f"{caution_detailed}"
    )

    with gzip.open(
        candidate_output,
        "wt",
        encoding="utf-8",
        newline="",
    ) as candidate_file:

        writer = csv.DictWriter(
            candidate_file,
            fieldnames=candidate_fields,
        )

        writer.writeheader()

        for problem_idx, (
            problem_id,
            problem,
        ) in enumerate(
            iter_detailed_json(
                caution_detailed
            ),
            start=1,
        ):

            problem_id = str(
                problem_id
            )

            detailed_ids.add(
                problem_id
            )

            if (
                problem_id
                not in input_metadata
            ):
                raise RuntimeError(
                    f"Caution problem ID "
                    f"not found in input: "
                    f"{problem_id}"
                )

            if (
                problem_id
                not in nb_results
            ):
                raise RuntimeError(
                    f"Caution problem ID "
                    f"not found in NeuBoots: "
                    f"{problem_id}"
                )

            meta = (
                input_metadata[
                    problem_id
                ]
            )

            nb = (
                nb_results[
                    problem_id
                ]
            )

            responses = problem[
                "responses"
            ]

            # --------------------------------------------
            # Candidate ordering
            # --------------------------------------------

            if isinstance(
                responses,
                dict,
            ):

                keys = sorted(
                    responses.keys(),
                    key=lambda x: int(x),
                )

                candidates = [
                    responses[key]
                    for key in keys
                ]

            elif isinstance(
                responses,
                list,
            ):

                candidates = responses

            else:
                raise TypeError(
                    f"{problem_id}: unsupported "
                    "responses container."
                )

            num_candidates = len(
                candidates
            )

            if (
                num_candidates
                != meta[
                    "num_candidates"
                ]
            ):
                raise RuntimeError(
                    f"{problem_id}: "
                    "Caution / input candidate "
                    "count mismatch."
                )

            if (
                num_candidates
                != len(
                    nb["uncertainty"]
                )
            ):
                raise RuntimeError(
                    f"{problem_id}: "
                    "Caution / NeuBoots candidate "
                    "count mismatch."
                )

            # --------------------------------------------
            # Strong ordering validation
            # --------------------------------------------

            detailed_samples = [
                candidate[
                    "content"
                ]
                for candidate
                in candidates
            ]

            detailed_digest = (
                candidate_digest(
                    detailed_samples
                )
            )

            if (
                detailed_digest
                != meta[
                    "candidate_digest"
                ]
            ):
                raise RuntimeError(
                    f"{problem_id}: "
                    "candidate ordering/content "
                    "does not match input JSONL."
                )

            # --------------------------------------------
            # Problem-level arrays
            # --------------------------------------------

            caution_values = []
            nb_values = []
            reward_values = []
            accuracy_values = []

            # --------------------------------------------
            # Candidate rows
            # --------------------------------------------

            for candidate_idx, candidate in enumerate(
                candidates,
                start=1,
            ):

                scores = (
                    candidate.get(
                        "reward_scores",
                        {},
                    )
                )

                reward_score = (
                    scores.get(
                        "reward_model_score"
                    )
                )

                rnd_score = (
                    scores.get(
                        "rnd_score"
                    )
                )

                accuracy = (
                    candidate.get(
                        "accuracy"
                    )
                )

                if reward_score is None:
                    raise RuntimeError(
                        f"{problem_id}, "
                        f"candidate {candidate_idx}: "
                        "missing reward_model_score."
                    )

                if rnd_score is None:
                    raise RuntimeError(
                        f"{problem_id}, "
                        f"candidate {candidate_idx}: "
                        "missing rnd_score."
                    )

                if accuracy is None:
                    raise RuntimeError(
                        f"{problem_id}, "
                        f"candidate {candidate_idx}: "
                        "missing accuracy."
                    )

                reward_score = float(
                    reward_score
                )

                rnd_score = float(
                    rnd_score
                )

                # Caution convention:
                #
                # rnd_score = - uncertainty
                #
                caution_uncertainty = (
                    -rnd_score
                )

                accuracy = (
                    to_accuracy(
                        accuracy
                    )
                )

                nb_mean = float(
                    nb["mean"][
                        candidate_idx - 1
                    ]
                )

                nb_uncertainty = float(
                    nb[
                        "uncertainty"
                    ][
                        candidate_idx - 1
                    ]
                )

                writer.writerow(
                    {
                        "dataset":
                            dataset_name,

                        "trial":
                            trial,

                        "problem_id":
                            problem_id,

                        "candidate_idx":
                            candidate_idx,

                        "prompt_char_length":
                            meta[
                                "prompt_char_length"
                            ],

                        "prompt_token_length":
                            meta[
                                "prompt_token_length"
                            ],

                        "reward_model_score":
                            reward_score,

                        "accuracy":
                            accuracy,

                        "caution_rnd_score":
                            rnd_score,

                        "caution_uncertainty":
                            caution_uncertainty,

                        "neuboots_mean":
                            nb_mean,

                        "neuboots_uncertainty":
                            nb_uncertainty,
                    }
                )

                caution_values.append(
                    caution_uncertainty
                )

                nb_values.append(
                    nb_uncertainty
                )

                reward_values.append(
                    reward_score
                )

                accuracy_values.append(
                    accuracy
                )

                num_candidate_rows += 1

            # --------------------------------------------
            # Problem-level aggregation
            # --------------------------------------------

            caution_values = np.asarray(
                caution_values,
                dtype=np.float64,
            )

            nb_values = np.asarray(
                nb_values,
                dtype=np.float64,
            )

            reward_values = np.asarray(
                reward_values,
                dtype=np.float64,
            )

            accuracy_values = np.asarray(
                accuracy_values,
                dtype=np.float64,
            )

            problem_rows.append(
                {
                    "dataset":
                        dataset_name,

                    "trial":
                        trial,

                    "problem_id":
                        problem_id,

                    "prompt_char_length":
                        meta[
                            "prompt_char_length"
                        ],

                    "prompt_token_length":
                        meta[
                            "prompt_token_length"
                        ],

                    "num_candidates":
                        num_candidates,

                    "caution_uncertainty_mean":
                        float(
                            caution_values.mean()
                        ),

                    "caution_uncertainty_median":
                        float(
                            np.median(
                                caution_values
                            )
                        ),

                    "caution_uncertainty_std":
                        float(
                            caution_values.std(
                                ddof=0
                            )
                        ),

                    "caution_uncertainty_max":
                        float(
                            caution_values.max()
                        ),

                    "neuboots_uncertainty_mean":
                        float(
                            nb_values.mean()
                        ),

                    "neuboots_uncertainty_median":
                        float(
                            np.median(
                                nb_values
                            )
                        ),

                    "neuboots_uncertainty_std":
                        float(
                            nb_values.std(
                                ddof=0
                            )
                        ),

                    "neuboots_uncertainty_max":
                        float(
                            nb_values.max()
                        ),

                    "reward_model_score_mean":
                        float(
                            reward_values.mean()
                        ),

                    "candidate_accuracy_mean":
                        float(
                            accuracy_values.mean()
                        ),
                }
            )

            if problem_idx % 100 == 0:
                print(
                    f"  processed "
                    f"{problem_idx} problems"
                )

    # ========================================================
    # 7. Final problem-ID validation
    # ========================================================

    if detailed_ids != input_ids:

        missing_detailed = (
            input_ids
            - detailed_ids
        )

        extra_detailed = (
            detailed_ids
            - input_ids
        )

        print()
        print(
            f"Missing detailed IDs: "
            f"{len(missing_detailed)}"
        )

        print(
            f"Extra detailed IDs: "
            f"{len(extra_detailed)}"
        )

        raise RuntimeError(
            "Caution detailed / input "
            "problem IDs do not match."
        )

    # ========================================================
    # 8. Save problem-level data
    # ========================================================

    problem_df = pd.DataFrame(
        problem_rows
    )

    problem_df.to_csv(
        problem_output,
        index=False,
    )

    # ========================================================
    # 9. Summary
    # ========================================================

    print()
    print("=" * 80)
    print("SUCCESS")
    print("=" * 80)

    print(
        f"Problems: "
        f"{len(problem_df)}"
    )

    print(
        f"Candidate rows: "
        f"{num_candidate_rows}"
    )

    print()
    print(
        f"Candidate-level output:\n"
        f"  {candidate_output}"
    )

    print()
    print(
        f"Problem-level output:\n"
        f"  {problem_output}"
    )

    print()
    print(
        "Prompt token length:"
    )

    print(
        problem_df[
            "prompt_token_length"
        ].describe()
    )

    print()
    print(
        "Mean Caution uncertainty:"
    )

    print(
        problem_df[
            "caution_uncertainty_mean"
        ].describe()
    )

    print()
    print(
        "Mean NeuBoots uncertainty:"
    )

    print(
        problem_df[
            "neuboots_uncertainty_mean"
        ].describe()
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "gsm8k",
            "math500",
            "bbh",
        ],
    )

    parser.add_argument(
        "--trial",
        required=True,
        type=int,
        choices=[
            1,
            2,
            3,
        ],
    )

    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--caution-detailed",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--neuboots-results",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=(
            "OpenAssistant/"
            "reward-model-deberta-v3-large-v2"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    extract(
        dataset_name=args.dataset,
        trial=args.trial,
        input_jsonl=args.input_jsonl,
        caution_detailed=args.caution_detailed,
        neuboots_results=args.neuboots_results,
        tokenizer_name=args.tokenizer_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()