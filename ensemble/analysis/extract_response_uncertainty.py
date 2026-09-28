#!/usr/bin/env python3

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import ijson
import numpy as np
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging


# Long sequences are intentionally tokenized without truncation
# to measure their original lengths.
hf_logging.set_verbosity_error()


# ============================================================
# Utility
# ============================================================

def candidate_key(value):
    """
    Sort Caution response keys such as:

        "1", "2", ..., "512"

    numerically.
    """

    text = str(value)

    numbers = re.findall(
        r"\d+",
        text,
    )

    if numbers:
        return int(numbers[-1]), text

    return 10**12, text


def candidate_digest(samples):
    """
    Hash the ordered candidate texts.

    Used to verify that:
        Caution candidate 1..512
    and
        NeuBoots input candidate 1..512

    are exactly the same responses in the same order.
    """

    h = hashlib.sha256()

    for sample in samples:

        if not isinstance(
            sample,
            str,
        ):
            sample = str(sample)

        h.update(
            sample.encode(
                "utf-8",
                errors="replace",
            )
        )

        h.update(
            b"\0"
        )

    return h.hexdigest()


def to_accuracy(value):

    if isinstance(
        value,
        bool,
    ):
        return float(value)

    if isinstance(
        value,
        (int, float),
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

        return float(value)

    raise TypeError(
        f"Unsupported accuracy value: {value}"
    )


def iter_detailed_json(path):
    """
    Stream the top-level Caution detailed JSON.
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
                f"Unsupported JSON root: {path}"
            )


# ============================================================
# NeuBoots input metadata
# ============================================================

def load_input_metadata(path):
    """
    Load the NeuBoots evaluation input.

    We mainly use this to verify:
        problem IDs
        candidate ordering/content
        prompt text
    """

    metadata = {}

    print()
    print(
        f"Loading evaluation input: {path}"
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

            item = json.loads(
                line
            )

            problem_id = str(
                item["instance_id"]
            )

            if problem_id in metadata:

                raise ValueError(
                    f"Duplicate instance_id: "
                    f"{problem_id}"
                )

            prompt = item[
                "prompt"
            ]

            samples = item[
                "all_samples"
            ]

            metadata[
                problem_id
            ] = {
                "prompt":
                    prompt,

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
            "No evaluation inputs loaded."
        )

    print(
        f"Evaluation input problems: "
        f"{len(metadata)}"
    )

    return metadata


# ============================================================
# NeuBoots results
# ============================================================

def load_neuboots_results(path):
    """
    Expected JSONL:

        instance_id
        all_neuboots_mc_mean
        all_neuboots_uncertainty
    """

    results = {}

    print()
    print(
        f"Loading NeuBoots scores: {path}"
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

            item = json.loads(
                line
            )

            problem_id = str(
                item["instance_id"]
            )

            if problem_id in results:

                raise ValueError(
                    f"Duplicate NeuBoots ID: "
                    f"{problem_id}"
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

                raise RuntimeError(
                    f"{problem_id}: "
                    "NeuBoots mean / uncertainty "
                    "length mismatch."
                )

            results[
                problem_id
            ] = {
                "mc_mean":
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
            "No NeuBoots results loaded."
        )

    print(
        f"NeuBoots problems: "
        f"{len(results)}"
    )

    return results


# ============================================================
# Length calculation
# ============================================================

def calculate_lengths(
        tokenizer,
        prompt,
        responses,
):
    """
    Compute several response/input length measures.

    response_char_length:
        Python len(response)

    response_word_length:
        whitespace-separated word count

    response_token_length:
        token count of RESPONSE ONLY
        without special tokens

    prompt_response_token_length:
        full tokenizer(prompt, response)
        length BEFORE truncation

    truncated_512:
        whether the actual scoring input would
        exceed max_length=512

    tokens_over_512:
        amount by which the pair exceeds 512
    """

    # --------------------------------------------------------
    # Character / word lengths
    # --------------------------------------------------------

    char_lengths = [
        len(response)
        for response
        in responses
    ]

    word_lengths = [
        len(
            response.split()
        )
        for response
        in responses
    ]

    # --------------------------------------------------------
    # Response-only tokens
    # --------------------------------------------------------

    response_encoded = tokenizer(
        responses,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )

    response_token_lengths = [
        len(ids)
        for ids
        in response_encoded[
            "input_ids"
        ]
    ]

    # --------------------------------------------------------
    # Prompt + response tokens
    #
    # This mirrors the type of pair passed to the RM /
    # NeuBoots scorer, but intentionally does NOT truncate.
    # --------------------------------------------------------

    prompts = [
        prompt
        for _ in responses
    ]

    pair_encoded = tokenizer(
        prompts,
        responses,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    pair_token_lengths = [
        len(ids)
        for ids
        in pair_encoded[
            "input_ids"
        ]
    ]

    truncated_512 = [
        int(length > 512)
        for length
        in pair_token_lengths
    ]

    tokens_over_512 = [
        max(
            0,
            length - 512,
        )
        for length
        in pair_token_lengths
    ]

    return (
        char_lengths,
        word_lengths,
        response_token_lengths,
        pair_token_lengths,
        truncated_512,
        tokens_over_512,
    )


# ============================================================
# Extraction
# ============================================================

def extract(
        dataset_name,
        trial,
        input_jsonl,
        caution_detailed,
        neuboots_results_path,
        tokenizer_name,
        output_path,
):

    print("=" * 80)
    print(
        "Response length / uncertainty extraction"
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
    # 2. Evaluation inputs
    # ========================================================

    input_metadata = (
        load_input_metadata(
            input_jsonl
        )
    )

    # ========================================================
    # 3. NeuBoots results
    # ========================================================

    nb_results = (
        load_neuboots_results(
            neuboots_results_path
        )
    )

    # ========================================================
    # 4. ID validation
    # ========================================================

    input_ids = set(
        input_metadata
    )

    nb_ids = set(
        nb_results
    )

    if input_ids != nb_ids:

        missing = (
            input_ids
            - nb_ids
        )

        extra = (
            nb_ids
            - input_ids
        )

        print(
            f"Missing NeuBoots IDs: "
            f"{len(missing)}"
        )

        print(
            f"Extra NeuBoots IDs: "
            f"{len(extra)}"
        )

        raise RuntimeError(
            "Evaluation input / NeuBoots "
            "problem IDs do not match."
        )

    # ========================================================
    # 5. Output setup
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "dataset",
        "trial",
        "problem_id",
        "candidate_idx",

        "response_char_length",
        "response_word_length",
        "response_token_length",

        "prompt_response_token_length",
        "truncated_512",
        "tokens_over_512",

        "reward_model_score",
        "accuracy",

        "caution_rnd_score",
        "caution_uncertainty",

        "neuboots_mean",
        "neuboots_uncertainty",

        "prediction_error",
        "absolute_prediction_error",
    ]

    detailed_ids = set()

    total_rows = 0

    truncated_rows = 0

    # ========================================================
    # 6. Stream Caution detailed candidates
    # ========================================================

    print()
    print(
        f"Reading Caution detailed: "
        f"{caution_detailed}"
    )

    with gzip.open(
        output_path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as out:

        writer = csv.DictWriter(
            out,
            fieldnames=fields,
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

            # --------------------------------------------
            # Problem alignment
            # --------------------------------------------

            if (
                problem_id
                not in input_metadata
            ):

                raise RuntimeError(
                    f"Caution problem missing "
                    f"from evaluation input: "
                    f"{problem_id}"
                )

            if (
                problem_id
                not in nb_results
            ):

                raise RuntimeError(
                    f"Caution problem missing "
                    f"from NeuBoots results: "
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

            prompt = meta[
                "prompt"
            ]

            responses_obj = problem[
                "responses"
            ]

            # --------------------------------------------
            # Preserve candidate ordering
            # --------------------------------------------

            if isinstance(
                responses_obj,
                dict,
            ):

                response_keys = sorted(
                    responses_obj.keys(),
                    key=candidate_key,
                )

                candidates = [
                    responses_obj[key]
                    for key
                    in response_keys
                ]

            elif isinstance(
                responses_obj,
                list,
            ):

                candidates = (
                    responses_obj
                )

            else:

                raise TypeError(
                    f"{problem_id}: unsupported "
                    "responses object."
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
                    "candidate count mismatch "
                    "between Caution and input."
                )

            if (
                num_candidates
                != len(
                    nb["uncertainty"]
                )
            ):

                raise RuntimeError(
                    f"{problem_id}: "
                    "candidate count mismatch "
                    "between Caution and NeuBoots."
                )

            # --------------------------------------------
            # Verify exact candidate text/order
            # --------------------------------------------

            responses = [
                candidate[
                    "content"
                ]
                for candidate
                in candidates
            ]

            if (
                candidate_digest(
                    responses
                )
                != meta[
                    "candidate_digest"
                ]
            ):

                raise RuntimeError(
                    f"{problem_id}: "
                    "candidate text/order does "
                    "not match evaluation input."
                )

            # --------------------------------------------
            # Calculate response lengths
            # --------------------------------------------

            (
                char_lengths,
                word_lengths,
                response_token_lengths,
                pair_token_lengths,
                truncated_flags,
                tokens_over_512,
            ) = calculate_lengths(
                tokenizer=tokenizer,
                prompt=prompt,
                responses=responses,
            )

            # --------------------------------------------
            # Candidate-level rows
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
                        "reward_model_score",
                        candidate.get(
                            "reward_model_score"
                        ),
                    )
                )

                rnd_score = (
                    scores.get(
                        "rnd_score",
                        candidate.get(
                            "rnd_score"
                        ),
                    )
                )

                accuracy = (
                    candidate.get(
                        "accuracy",
                        candidate.get(
                            "correct",
                            candidate.get(
                                "is_correct"
                            ),
                        ),
                    )
                )

                if reward_score is None:

                    raise RuntimeError(
                        f"{problem_id}, "
                        f"candidate {candidate_idx}: "
                        "missing reward_model_score"
                    )

                if rnd_score is None:

                    raise RuntimeError(
                        f"{problem_id}, "
                        f"candidate {candidate_idx}: "
                        "missing rnd_score"
                    )

                if accuracy is None:

                    raise RuntimeError(
                        f"{problem_id}, "
                        f"candidate {candidate_idx}: "
                        "missing accuracy"
                    )

                reward_score = float(
                    reward_score
                )

                rnd_score = float(
                    rnd_score
                )

                accuracy = to_accuracy(
                    accuracy
                )

                # ----------------------------------------
                # Caution:
                #
                # stored rnd_score = - uncertainty
                # ----------------------------------------

                caution_uncertainty = (
                    -rnd_score
                )

                # ----------------------------------------
                # NeuBoots
                # ----------------------------------------

                array_idx = (
                    candidate_idx
                    - 1
                )

                nb_mean = float(
                    nb[
                        "mc_mean"
                    ][
                        array_idx
                    ]
                )

                nb_uncertainty = float(
                    nb[
                        "uncertainty"
                    ][
                        array_idx
                    ]
                )

                # ----------------------------------------
                # Useful for later analysis:
                #
                # RM target - ensemble mean prediction
                # ----------------------------------------

                prediction_error = (
                    nb_mean
                    - reward_score
                )

                absolute_prediction_error = abs(
                    prediction_error
                )

                truncated = (
                    truncated_flags[
                        array_idx
                    ]
                )

                truncated_rows += (
                    truncated
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

                        "response_char_length":
                            char_lengths[
                                array_idx
                            ],

                        "response_word_length":
                            word_lengths[
                                array_idx
                            ],

                        "response_token_length":
                            response_token_lengths[
                                array_idx
                            ],

                        "prompt_response_token_length":
                            pair_token_lengths[
                                array_idx
                            ],

                        "truncated_512":
                            truncated,

                        "tokens_over_512":
                            tokens_over_512[
                                array_idx
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

                        "prediction_error":
                            prediction_error,

                        "absolute_prediction_error":
                            absolute_prediction_error,
                    }
                )

                total_rows += 1

            if problem_idx % 100 == 0:

                print(
                    f"  processed "
                    f"{problem_idx} problems"
                )

    # ========================================================
    # 7. Final ID check
    # ========================================================

    if detailed_ids != input_ids:

        missing = (
            input_ids
            - detailed_ids
        )

        extra = (
            detailed_ids
            - input_ids
        )

        print()
        print(
            f"Missing detailed IDs: "
            f"{len(missing)}"
        )

        print(
            f"Extra detailed IDs: "
            f"{len(extra)}"
        )

        raise RuntimeError(
            "Caution detailed / evaluation "
            "input IDs do not match."
        )

    # ========================================================
    # 8. Summary
    # ========================================================

    print()
    print("=" * 80)
    print("SUCCESS")
    print("=" * 80)

    print(
        f"Problems: "
        f"{len(detailed_ids)}"
    )

    print(
        f"Candidate rows: "
        f"{total_rows}"
    )

    print(
        f"Inputs exceeding 512 tokens: "
        f"{truncated_rows}"
    )

    if total_rows > 0:

        print(
            "Truncation ratio: "
            f"{100.0 * truncated_rows / total_rows:.2f}%"
        )

    print()
    print(
        f"Saved: {output_path}"
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
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    extract(
        dataset_name=args.dataset,
        trial=args.trial,
        input_jsonl=args.input_jsonl,
        caution_detailed=args.caution_detailed,
        neuboots_results_path=(
            args.neuboots_results
        ),
        tokenizer_name=args.tokenizer_name,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()