import os
import json
import argparse
import logging
import random

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer

from ensemble.models.neuboots_reward_model import (
    RewardValuePredictor,
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
# Quantile settings
#
# For M MC predictions:
#
#     k_alpha = floor(alpha * M)
#
# We directly use the sorted-array index k_alpha.
# This is NOT torch.quantile interpolation.
# ============================================================

ALPHAS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
)


# ============================================================
# Seed
# ============================================================

def set_seed(
        seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Predictor loading
# ============================================================

def load_predictor(
        checkpoint_dir: str,
        device: str,
):

    config_path = os.path.join(
        checkpoint_dir,
        "reward_predictor_config.json",
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "reward_predictor.pt",
    )

    if not os.path.exists(
        config_path
    ):
        raise FileNotFoundError(
            f"Config not found: "
            f"{config_path}"
        )

    if not os.path.exists(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    reward_model_path = (
        config[
            "reward_model_path"
        ]
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            reward_model_path,
            use_fast=True,
            trust_remote_code=True,
        )
    )

    # --------------------------------------------------------
    # Predictor architecture
    # --------------------------------------------------------

    predictor = RewardValuePredictor(
        model_path=(
            reward_model_path
        ),
        num_layers=(
            config[
                "predictor_layers"
            ]
        ),
        exact_architecture=(
            config[
                "exact_architecture"
            ]
        ),
        embedding_strategy=(
            config[
                "embedding_strategy"
            ]
        ),
        use_projection=(
            config[
                "use_projection"
            ]
        ),
    ).to(device)

    # --------------------------------------------------------
    # Load predictor weights
    # --------------------------------------------------------

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    predictor.load_state_dict(
        state_dict
    )

    predictor.eval()

    logger.info(
        "Loaded NeuBoots predictor from "
        f"{checkpoint_path}"
    )

    return (
        tokenizer,
        predictor,
    )


# ============================================================
# NeuBoots scoring
# ============================================================

@torch.inference_mode()
def score_batch(
        tokenizer,
        predictor,
        prompt: str,
        responses,
        num_mc: int,
        device: str,
        alphas=ALPHAS,
):
    """
    Compute NeuBoots MC predictions and uncertainty scores.

    Parameters
    ----------
    reward_samples:
        M NeuBoots stochastic reward predictions

        shape:
            [M, batch_size]

    Let the sorted predictions be

        a_0 <= a_1 <= ... <= a_{M-1}

    and

        k_alpha = floor(alpha * M)

    We compute:

    1. STD

        std(r_1, ..., r_M)

    2. Distance

        NOT computed here.

        It requires the main reward model score:

            |r_RM - mean(r_1, ..., r_M)|

        Therefore we save mc_mean and compute distance
        later during calibration/evaluation.

    3. Median - Quantile

        a_floor(M/2) - a_floor(alpha*M)

    4. Average - Quantile

        mean(r_1, ..., r_M)
        - a_floor(alpha*M)

    5. Average - Quantile's Average

        mean(r_1, ..., r_M)
        - mean(
            a_0,
            ...,
            a_floor(alpha*M)
        )
    """

    if num_mc <= 0:
        raise ValueError(
            "num_mc must be positive, "
            f"got {num_mc}"
        )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    prompts = [
        prompt
    ] * len(
        responses
    )

    inputs = tokenizer(
        prompts,
        responses,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = (
        inputs[
            "input_ids"
        ].to(device)
    )

    attention_mask = (
        inputs[
            "attention_mask"
        ].to(device)
    )

    # --------------------------------------------------------
    # MC NeuBoots reward predictions
    #
    # reward_samples:
    #
    #     [M, batch_size]
    # --------------------------------------------------------

    reward_samples = predictor(
        input_ids,
        attention_mask,
        alpha=num_mc,
    )

    if reward_samples.dim() != 2:
        raise RuntimeError(
            "Expected reward_samples shape "
            "[num_mc, batch_size], got "
            f"{tuple(reward_samples.shape)}"
        )

    M = int(
        reward_samples.shape[0]
    )

    if M != num_mc:
        raise RuntimeError(
            f"Expected {num_mc} MC samples, "
            f"got {M}"
        )

    # ========================================================
    # Ensemble mean
    #
    # Needed both as a statistic and later for:
    #
    #     distance = |RM - ensemble mean|
    # ========================================================

    mc_mean = (
        reward_samples.mean(
            dim=0
        )
    )

    # ========================================================
    # 1. Standard deviation
    # ========================================================

    mc_std = (
        reward_samples.std(
            dim=0,
            unbiased=False,
        )
    )

    # ========================================================
    # Sort MC reward predictions
    #
    # For every candidate:
    #
    #     a_0 <= a_1 <= ... <= a_{M-1}
    #
    # shape remains:
    #
    #     [M, batch_size]
    # ========================================================

    sorted_samples = (
        torch.sort(
            reward_samples,
            dim=0,
        ).values
    )

    # --------------------------------------------------------
    # Center point
    #
    # User-defined:
    #
    #     arr[floor(M / 2)]
    #
    # For M=20:
    #
    #     floor(20 / 2) = 10
    #
    # so we use a_10 with zero-based indexing.
    #
    # This is intentionally NOT the conventional
    # even-sample statistical median:
    #
    #     (a_9 + a_10) / 2
    # --------------------------------------------------------

    median_index = (
        M // 2
    )

    median_point = (
        sorted_samples[
            median_index
        ]
    )

    # ========================================================
    # Containers for alpha-dependent uncertainty scores
    # ========================================================

    median_quantile = {}

    avg_quantile = {}

    avg_quantile_avg = {}

    # ========================================================
    # Alpha sweep
    # ========================================================

    for alpha in alphas:

        if not (
            0.0
            <= alpha
            < 0.5
        ):
            raise ValueError(
                "alpha must satisfy "
                "0 <= alpha < 0.5, "
                f"got {alpha}"
            )

        # ----------------------------------------------------
        # k_alpha = floor(alpha * M)
        #
        # Example M=20:
        #
        # alpha=0.0 -> k=0
        # alpha=0.1 -> k=2
        # alpha=0.2 -> k=4
        # alpha=0.3 -> k=6
        # alpha=0.4 -> k=8
        # ----------------------------------------------------

        k_alpha = int(
            np.floor(
                alpha * M
            )
        )

        if (
            k_alpha < 0
            or k_alpha >= M
        ):
            raise RuntimeError(
                "Invalid quantile index: "
                f"alpha={alpha}, "
                f"M={M}, "
                f"k_alpha={k_alpha}"
            )

        # ----------------------------------------------------
        # Single lower quantile point
        #
        #     a_floor(alpha*M)
        # ----------------------------------------------------

        quantile_point = (
            sorted_samples[
                k_alpha
            ]
        )

        # ----------------------------------------------------
        # Lower quantile average
        #
        #     1 / (k_alpha + 1)
        #     * sum_{j=0}^{k_alpha} a_j
        #
        # Note:
        #
        #     slicing endpoint is exclusive,
        #     therefore use :k_alpha + 1
        # ----------------------------------------------------

        lower_quantile_avg = (
            sorted_samples[
                :k_alpha + 1
            ]
            .mean(
                dim=0
            )
        )

        alpha_key = (
            f"{alpha:.1f}"
        )

        # ====================================================
        # 3. Median - Quantile
        #
        #     a_floor(M/2)
        #     -
        #     a_floor(alpha*M)
        # ====================================================

        median_quantile[
            alpha_key
        ] = (
            median_point
            - quantile_point
        ).cpu().tolist()

        # ====================================================
        # 4. Average - Quantile
        #
        #     mean(r_1, ..., r_M)
        #     -
        #     a_floor(alpha*M)
        # ====================================================

        avg_quantile[
            alpha_key
        ] = (
            mc_mean
            - quantile_point
        ).cpu().tolist()

        # ====================================================
        # 5. Average - Quantile's Average
        #
        #     mean(r_1, ..., r_M)
        #     -
        #     mean(
        #         a_0,
        #         ...,
        #         a_floor(alpha*M)
        #     )
        # ====================================================

        avg_quantile_avg[
            alpha_key
        ] = (
            mc_mean
            - lower_quantile_avg
        ).cpu().tolist()

    # ========================================================
    # Return
    # ========================================================

    return (
        mc_mean.cpu().tolist(),
        mc_std.cpu().tolist(),
        median_quantile,
        avg_quantile,
        avg_quantile_avg,
    )


# ============================================================
# Resume support
# ============================================================

def load_completed_ids(
        output_path: str,
):

    completed = set()

    if not os.path.exists(
        output_path
    ):
        return completed

    with open(
        output_path,
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

            completed.add(
                item[
                    "instance_id"
                ]
            )

    return completed


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Input / model / output
    # --------------------------------------------------------

    parser.add_argument(
        "--caution-results",
        type=str,
        required=True,
        help=(
            "Prepared JSONL containing "
            "instance_id, prompt, and all_samples."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help=(
            "NeuBoots predictor checkpoint directory."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # MC / batching
    # --------------------------------------------------------

    parser.add_argument(
        "--num-mc",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
    )

    # --------------------------------------------------------
    # Device / seed
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

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if args.num_mc <= 0:
        raise ValueError(
            "--num-mc must be positive, "
            f"got {args.num_mc}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive, "
            f"got {args.batch_size}"
        )

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        args.seed
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = os.path.dirname(
        args.output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # ========================================================
    # Load NeuBoots predictor
    # ========================================================

    (
        tokenizer,
        predictor,
    ) = load_predictor(
        checkpoint_dir=(
            args.checkpoint_dir
        ),
        device=(
            args.device
        ),
    )

    # ========================================================
    # Resume previously completed problem IDs
    # ========================================================

    completed_ids = (
        load_completed_ids(
            args.output_path
        )
    )

    if completed_ids:

        logger.info(
            f"Found "
            f"{len(completed_ids)} "
            f"already scored problems"
        )

    processed = 0

    # ========================================================
    # Score problems
    # ========================================================

    with open(
        args.caution_results,
        "r",
        encoding="utf-8",
    ) as input_file:

        for line in tqdm(
            input_file,
            desc=(
                "Scoring NeuBoots candidates"
            ),
        ):

            line = (
                line.strip()
            )

            if not line:
                continue

            item = json.loads(
                line
            )

            instance_id = (
                item[
                    "instance_id"
                ]
            )

            # ------------------------------------------------
            # Resume
            # ------------------------------------------------

            if (
                instance_id
                in completed_ids
            ):
                continue

            if (
                args.max_problems
                is not None
                and processed
                >= args.max_problems
            ):
                break

            # ------------------------------------------------
            # Candidate input
            # ------------------------------------------------

            prompt = (
                item[
                    "prompt"
                ]
            )

            responses = (
                item[
                    "all_samples"
                ]
            )

            if (
                len(responses)
                != 512
            ):
                logger.warning(
                    f"{instance_id}: "
                    f"expected 512 responses, "
                    f"got {len(responses)}"
                )

            # =================================================
            # Candidate-level result containers
            # =================================================

            all_mc_mean = []

            all_uncertainty_std = []

            all_uncertainty_median_quantile = {
                f"{alpha:.1f}": []
                for alpha
                in ALPHAS
            }

            all_uncertainty_avg_quantile = {
                f"{alpha:.1f}": []
                for alpha
                in ALPHAS
            }

            all_uncertainty_avg_quantile_avg = {
                f"{alpha:.1f}": []
                for alpha
                in ALPHAS
            }

            # =================================================
            # Batch candidates
            # =================================================

            for start in range(
                0,
                len(responses),
                args.batch_size,
            ):

                end = min(
                    start
                    + args.batch_size,
                    len(responses),
                )

                batch_responses = (
                    responses[
                        start:end
                    ]
                )

                (
                    mc_mean,
                    mc_std,
                    median_quantile,
                    avg_quantile,
                    avg_quantile_avg,
                ) = score_batch(
                    tokenizer=(
                        tokenizer
                    ),
                    predictor=(
                        predictor
                    ),
                    prompt=(
                        prompt
                    ),
                    responses=(
                        batch_responses
                    ),
                    num_mc=(
                        args.num_mc
                    ),
                    device=(
                        args.device
                    ),
                    alphas=(
                        ALPHAS
                    ),
                )

                # --------------------------------------------
                # Mean
                # --------------------------------------------

                all_mc_mean.extend(
                    mc_mean
                )

                # --------------------------------------------
                # 1. STD
                # --------------------------------------------

                all_uncertainty_std.extend(
                    mc_std
                )

                # --------------------------------------------
                # Alpha-dependent scores
                # --------------------------------------------

                for alpha in ALPHAS:

                    alpha_key = (
                        f"{alpha:.1f}"
                    )

                    # 3. Median - Quantile
                    all_uncertainty_median_quantile[
                        alpha_key
                    ].extend(
                        median_quantile[
                            alpha_key
                        ]
                    )

                    # 4. Average - Quantile
                    all_uncertainty_avg_quantile[
                        alpha_key
                    ].extend(
                        avg_quantile[
                            alpha_key
                        ]
                    )

                    # 5. Average - Quantile's Average
                    all_uncertainty_avg_quantile_avg[
                        alpha_key
                    ].extend(
                        avg_quantile_avg[
                            alpha_key
                        ]
                    )

            # =================================================
            # Output record
            # =================================================

            result = {
                "instance_id":
                    instance_id,

                "num_candidates":
                    len(
                        responses
                    ),

                "num_mc":
                    args.num_mc,

                "alphas":
                    list(
                        ALPHAS
                    ),

                # --------------------------------------------
                # Ensemble mean
                #
                # Later used for:
                #
                # distance =
                # |main_rm - ensemble_mean|
                # --------------------------------------------

                "all_neuboots_mc_mean":
                    all_mc_mean,

                # --------------------------------------------
                # 1. STD
                # --------------------------------------------

                "all_neuboots_uncertainty_std":
                    all_uncertainty_std,

                # --------------------------------------------
                # Backward compatibility
                #
                # Existing scripts historically interpret
                # all_neuboots_uncertainty as STD.
                # --------------------------------------------

                "all_neuboots_uncertainty":
                    all_uncertainty_std,

                # --------------------------------------------
                # 3. Median - Quantile
                # --------------------------------------------

                "all_neuboots_uncertainty_median_quantile":
                    (
                        all_uncertainty_median_quantile
                    ),

                # --------------------------------------------
                # 4. Average - Quantile
                # --------------------------------------------

                "all_neuboots_uncertainty_avg_quantile":
                    (
                        all_uncertainty_avg_quantile
                    ),

                # --------------------------------------------
                # 5. Average - Quantile's Average
                # --------------------------------------------

                "all_neuboots_uncertainty_avg_quantile_avg":
                    (
                        all_uncertainty_avg_quantile_avg
                    ),
            }

            # =================================================
            # Append one problem
            # =================================================

            with open(
                args.output_path,
                "a",
                encoding="utf-8",
            ) as output_file:

                output_file.write(
                    json.dumps(
                        result
                    )
                    + "\n"
                )

            processed += 1

    # ========================================================
    # Done
    # ========================================================

    logger.info(
        f"Finished scoring "
        f"{processed} problems"
    )


if __name__ == "__main__":
    main()