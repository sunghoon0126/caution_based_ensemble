import os
import gc
import json
import argparse
import logging
import random

import numpy as np
import torch
from tqdm import tqdm

from baseline.pessimism.datasets.gsm8k import (
    GSM8KDataset,
)
from baseline.pessimism.models.openai_model import (
    run_openai_inference,
)
from baseline.pessimism.models.rnd_reward_model import (
    RNDRewardModel,
)

from ensemble.evaluation.score_gsm8k_neuboots import (
    ALPHAS,
    load_predictor,
    score_batch,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Calibration response generation
#
# Normally we will NOT use this for the current experiment,
# because the existing held-out GSM8K responses are reused.
# ============================================================

def load_calibration_dataset(
        start_index,
        num_examples,
        seed,
):
    prompt_postprocessor_config = {
        "system_prompt": (
            "Solve the following problem step by step. "
            "Give your final numerical answer at the end with: "
            "#### {NUM}"
        ),
        "add_generation_prompt": True,
    }

    dataset = GSM8KDataset(
        seed=seed,
        split="train",
        name_or_path="gsm8k",
        config_name="main",
        fewshot_num=0,
        prompt_postprocessor_config=(
            prompt_postprocessor_config
        ),
    )

    end_index = (
        start_index
        + num_examples
    )

    if end_index > len(dataset):
        raise ValueError(
            f"Requested slice "
            f"[{start_index}:{end_index}], "
            f"but GSM8K train has "
            f"{len(dataset)} examples."
        )

    dataset.problems = (
        dataset.problems[
            start_index:end_index
        ]
    )

    logger.info(
        f"Calibration slice: "
        f"GSM8K train "
        f"[{start_index}:{end_index}]"
    )

    logger.info(
        f"Calibration problems: "
        f"{len(dataset)}"
    )

    return dataset


def generate_responses(
        dataset,
        inference_config,
        output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    requests = []

    for local_idx, problem in enumerate(
        dataset
    ):
        request_uuid = (
            f"gsm8k_calibration_"
            f"{local_idx}"
        )

        requests.append(
            {
                "uuid":
                    request_uuid,

                "prompt":
                    problem.prompt,

                "messages": [
                    {
                        "role": "user",
                        "content": problem.prompt,
                    }
                ],
            }
        )

    inference_kwargs = (
        inference_config
        .get(
            "inference_kwargs",
            {},
        )
        .copy()
    )

    inference_kwargs[
        "output_path"
    ] = output_dir

    logger.info(
        f"Generating "
        f"{len(requests)} "
        f"calibration responses"
    )

    run_openai_inference(
        requests=requests,
        **inference_kwargs,
    )

    responses_file = os.path.join(
        output_dir,
        "all_responses.jsonl",
    )

    if not os.path.exists(
        responses_file
    ):
        raise FileNotFoundError(
            responses_file
        )

    return responses_file


# ============================================================
# Existing response loader
# ============================================================

def extract_prompt_response(item):
    request = item.get(
        "request",
        {},
    )

    response_obj = item.get(
        "response",
        {},
    )

    prompt = request.get(
        "prompt"
    )

    response = response_obj.get(
        "generated_text"
    )

    if (
        not isinstance(
            prompt,
            str,
        )
        or not isinstance(
            response,
            str,
        )
    ):
        return None

    return (
        prompt,
        response,
    )


def load_pairs(path):
    prompts = []
    responses = []

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

            pair = (
                extract_prompt_response(
                    item
                )
            )

            if pair is None:
                continue

            (
                prompt,
                response,
            ) = pair

            prompts.append(
                prompt
            )

            responses.append(
                response
            )

    if not prompts:
        raise RuntimeError(
            "No calibration "
            "prompt-response pairs found."
        )

    logger.info(
        f"Loaded "
        f"{len(prompts)} "
        f"calibration pairs"
    )

    return (
        prompts,
        responses,
    )


# ============================================================
# Main RM + original Caution RND
# ============================================================

def score_original_rnd(
        prompts,
        responses,
        reward_model_path,
        rnd_model_path,
        device,
):
    config_path = os.path.join(
        rnd_model_path,
        "rnd_config.json",
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    model = RNDRewardModel(
        reward_model_path=(
            reward_model_path
        ),
        target_layers=(
            config[
                "target_layers"
            ]
        ),
        predictor_layers=(
            config[
                "predictor_layers"
            ]
        ),
        rnd_weight=(
            config.get(
                "rnd_weight",
                0.2,
            )
        ),
        device=device,
        exact_architecture=(
            config.get(
                "exact_architecture",
                False,
            )
        ),
        embedding_strategy=(
            config.get(
                "embedding_strategy",
                "shared_trainable",
            )
        ),
        use_projection=(
            config.get(
                "use_projection",
                True,
            )
        ),
    )

    model.load_predictor(
        rnd_model_path
    )

    reward_scores = []
    rnd_uncertainties = []

    for (
        prompt,
        response,
    ) in tqdm(
        zip(
            prompts,
            responses,
        ),
        total=len(prompts),
        desc="RM + RND calibration scoring",
    ):
        reward_score = (
            model.compute_reward_score(
                prompt,
                response,
            )
        )

        rnd_score = (
            model.compute_rnd_score(
                prompt,
                response,
            )
        )

        # Caution convention:
        #
        # rnd_score = -MSE
        #
        # Convert to positive uncertainty.
        rnd_uncertainty = (
            -rnd_score
        )

        reward_scores.append(
            float(
                reward_score
            )
        )

        rnd_uncertainties.append(
            float(
                rnd_uncertainty
            )
        )

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        reward_scores,
        rnd_uncertainties,
    )


# ============================================================
# NeuBoots calibration scoring
#
# Returns:
#
#   mc_mean
#   std
#
#   median_quantile[alpha]
#   avg_quantile[alpha]
#   avg_quantile_avg[alpha]
#
# Distance is computed later using:
#
#   |main RM - mc_mean|
# ============================================================

def score_neuboots(
        prompts,
        responses,
        checkpoint_dir,
        num_mc,
        device,
):
    (
        tokenizer,
        predictor,
    ) = load_predictor(
        checkpoint_dir=(
            checkpoint_dir
        ),
        device=device,
    )

    mc_means = []

    std_values = []

    median_quantile_values = {
        f"{alpha:.1f}": []
        for alpha in ALPHAS
    }

    avg_quantile_values = {
        f"{alpha:.1f}": []
        for alpha in ALPHAS
    }

    avg_quantile_avg_values = {
        f"{alpha:.1f}": []
        for alpha in ALPHAS
    }

    # score_batch() assumes one shared prompt
    # for all responses in a batch.
    #
    # Calibration pairs have different prompts,
    # so score them one pair at a time.
    for (
        prompt,
        response,
    ) in tqdm(
        zip(
            prompts,
            responses,
        ),
        total=len(prompts),
        desc="NeuBoots calibration scoring",
    ):
        (
            mc_mean,
            mc_std,
            median_quantile,
            avg_quantile,
            avg_quantile_avg,
        ) = score_batch(
            tokenizer=tokenizer,
            predictor=predictor,
            prompt=prompt,
            responses=[
                response
            ],
            num_mc=num_mc,
            device=device,
            alphas=ALPHAS,
        )

        # One response was passed,
        # therefore every returned list
        # contains exactly one value.

        mc_means.append(
            float(
                mc_mean[0]
            )
        )

        std_values.append(
            float(
                mc_std[0]
            )
        )

        for alpha in ALPHAS:

            alpha_key = (
                f"{alpha:.1f}"
            )

            median_quantile_values[
                alpha_key
            ].append(
                float(
                    median_quantile[
                        alpha_key
                    ][0]
                )
            )

            avg_quantile_values[
                alpha_key
            ].append(
                float(
                    avg_quantile[
                        alpha_key
                    ][0]
                )
            )

            avg_quantile_avg_values[
                alpha_key
            ].append(
                float(
                    avg_quantile_avg[
                        alpha_key
                    ][0]
                )
            )

    del predictor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mc_mean":
            mc_means,

        "std":
            std_values,

        "median_quantile":
            median_quantile_values,

        "avg_quantile":
            avg_quantile_values,

        "avg_quantile_avg":
            avg_quantile_avg_values,
    }


# ============================================================
# Distance
#
#     |r_RM - ensemble mean|
# ============================================================

def compute_distance(
        reward_scores,
        mc_means,
):
    reward_array = np.asarray(
        reward_scores,
        dtype=np.float64,
    )

    mean_array = np.asarray(
        mc_means,
        dtype=np.float64,
    )

    if (
        reward_array.shape
        != mean_array.shape
    ):
        raise ValueError(
            "Reward / NeuBoots mean "
            "shape mismatch: "
            f"{reward_array.shape} "
            f"vs "
            f"{mean_array.shape}"
        )

    distance = np.abs(
        reward_array
        - mean_array
    )

    return (
        distance.tolist()
    )


# ============================================================
# Save candidate-level calibration scores
# ============================================================

def save_calibration_scores(
        output_path,
        prompts,
        responses,
        reward_scores,
        rnd_uncertainties,
        neuboots_scores,
        distance_values,
):
    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    num_examples = len(
        prompts
    )

    fields_to_check = {
        "responses":
            responses,

        "reward_scores":
            reward_scores,

        "rnd_uncertainties":
            rnd_uncertainties,

        "mc_mean":
            neuboots_scores[
                "mc_mean"
            ],

        "std":
            neuboots_scores[
                "std"
            ],

        "distance":
            distance_values,
    }

    for (
        name,
        values,
    ) in fields_to_check.items():

        if (
            len(values)
            != num_examples
        ):
            raise ValueError(
                f"{name} length "
                f"{len(values)} "
                f"!= "
                f"{num_examples}"
            )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for i in range(
            num_examples
        ):
            median_quantile = {}

            avg_quantile = {}

            avg_quantile_avg = {}

            for alpha in ALPHAS:

                alpha_key = (
                    f"{alpha:.1f}"
                )

                median_quantile[
                    alpha_key
                ] = float(
                    neuboots_scores[
                        "median_quantile"
                    ][
                        alpha_key
                    ][i]
                )

                avg_quantile[
                    alpha_key
                ] = float(
                    neuboots_scores[
                        "avg_quantile"
                    ][
                        alpha_key
                    ][i]
                )

                avg_quantile_avg[
                    alpha_key
                ] = float(
                    neuboots_scores[
                        "avg_quantile_avg"
                    ][
                        alpha_key
                    ][i]
                )

            item = {
                "calibration_index":
                    i,

                "prompt":
                    prompts[i],

                "response":
                    responses[i],

                # --------------------------------------------
                # Main RM
                # --------------------------------------------

                "reward_score":
                    float(
                        reward_scores[i]
                    ),

                # --------------------------------------------
                # Original Caution uncertainty
                # --------------------------------------------

                "rnd_uncertainty":
                    float(
                        rnd_uncertainties[i]
                    ),

                # --------------------------------------------
                # NeuBoots ensemble mean
                # --------------------------------------------

                "neuboots_mc_mean":
                    float(
                        neuboots_scores[
                            "mc_mean"
                        ][i]
                    ),

                # --------------------------------------------
                # 1. STD
                # --------------------------------------------

                "neuboots_uncertainty_std":
                    float(
                        neuboots_scores[
                            "std"
                        ][i]
                    ),

                # Backward compatibility:
                # old code interprets this as STD.
                "neuboots_uncertainty":
                    float(
                        neuboots_scores[
                            "std"
                        ][i]
                    ),

                # --------------------------------------------
                # 2. Distance
                #
                # |RM - ensemble mean|
                # --------------------------------------------

                "neuboots_uncertainty_distance":
                    float(
                        distance_values[i]
                    ),

                # --------------------------------------------
                # 3. Median - Quantile
                # --------------------------------------------

                "neuboots_uncertainty_median_quantile":
                    median_quantile,

                # --------------------------------------------
                # 4. Average - Quantile
                # --------------------------------------------

                "neuboots_uncertainty_avg_quantile":
                    avg_quantile,

                # --------------------------------------------
                # 5. Average - Quantile's Average
                # --------------------------------------------

                "neuboots_uncertainty_avg_quantile_avg":
                    avg_quantile_avg,
            }

            f.write(
                json.dumps(
                    item
                )
                + "\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Only needed when generating a new calibration set.
    # For the current experiment we reuse --responses-file.
    # --------------------------------------------------------

    parser.add_argument(
        "--start-index",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--inference-config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--generation-output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--responses-file",
        type=str,
        default=None,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=(
            "OpenAssistant/"
            "reward-model-deberta-v3-large-v2"
        ),
    )

    parser.add_argument(
        "--rnd-model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--neuboots-checkpoint-dir",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # NeuBoots MC
    # --------------------------------------------------------

    parser.add_argument(
        "--num-mc",
        type=int,
        default=20,
    )

    # Kept for CLI compatibility.
    #
    # Current calibration implementation scores
    # one prompt-response pair at a time because
    # score_batch() assumes a shared prompt.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.num_mc <= 0:
        raise ValueError(
            "--num-mc must be positive."
        )

    set_seed(
        args.seed
    )

    # ========================================================
    # 1. Reuse existing calibration responses
    #    or generate new ones if explicitly requested.
    # ========================================================

    responses_file = (
        args.responses_file
    )

    if responses_file is None:

        if args.inference_config is None:
            raise ValueError(
                "--inference-config is required "
                "when --responses-file is not provided."
            )

        if (
            args.generation_output_dir
            is None
        ):
            raise ValueError(
                "--generation-output-dir is required "
                "when --responses-file is not provided."
            )

        dataset = (
            load_calibration_dataset(
                start_index=(
                    args.start_index
                ),
                num_examples=(
                    args.num_examples
                ),
                seed=args.seed,
            )
        )

        with open(
            args.inference_config,
            "r",
            encoding="utf-8",
        ) as f:

            inference_config = (
                json.load(f)
            )

        responses_file = (
            generate_responses(
                dataset=dataset,
                inference_config=(
                    inference_config
                ),
                output_dir=(
                    args.generation_output_dir
                ),
            )
        )

    # ========================================================
    # 2. Load calibration prompt-response pairs
    # ========================================================

    (
        prompts,
        responses,
    ) = load_pairs(
        responses_file
    )

    # ========================================================
    # 3. Main RM + original RND
    # ========================================================

    (
        reward_scores,
        rnd_uncertainties,
    ) = score_original_rnd(
        prompts=prompts,
        responses=responses,
        reward_model_path=(
            args.reward_model_path
        ),
        rnd_model_path=(
            args.rnd_model_path
        ),
        device=args.device,
    )

    # ========================================================
    # 4. NeuBoots
    # ========================================================

    neuboots_scores = (
        score_neuboots(
            prompts=prompts,
            responses=responses,
            checkpoint_dir=(
                args.neuboots_checkpoint_dir
            ),
            num_mc=(
                args.num_mc
            ),
            device=(
                args.device
            ),
        )
    )

    # ========================================================
    # 5. Distance
    #
    # |main RM - NeuBoots ensemble mean|
    # ========================================================

    distance_values = (
        compute_distance(
            reward_scores=(
                reward_scores
            ),
            mc_means=(
                neuboots_scores[
                    "mc_mean"
                ]
            ),
        )
    )

    # ========================================================
    # 6. Save
    # ========================================================

    save_calibration_scores(
        output_path=(
            args.output_path
        ),
        prompts=prompts,
        responses=responses,
        reward_scores=(
            reward_scores
        ),
        rnd_uncertainties=(
            rnd_uncertainties
        ),
        neuboots_scores=(
            neuboots_scores
        ),
        distance_values=(
            distance_values
        ),
    )

    print()
    print(
        "Saved calibration scores:"
    )

    print(
        f"  {args.output_path}"
    )

    print(
        f"Number of calibration pairs: "
        f"{len(prompts)}"
    )

    print(
        "NeuBoots metrics:"
    )

    print(
        "  1. std"
    )

    print(
        "  2. distance"
    )

    print(
        "  3. median_quantile"
    )

    print(
        "  4. avg_quantile"
    )

    print(
        "  5. avg_quantile_avg"
    )

    print(
        "alphas:",
        [
            f"{alpha:.1f}"
            for alpha in ALPHAS
        ],
    )


if __name__ == "__main__":
    main()