import os
import argparse
import logging
import json
import torch
import numpy as np
import random
from typing import List, Dict, Optional, Tuple

# Import the reward model
from ensemble.models.neuboots_reward_model import RewardValueModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_prompt_response(
        item: Dict,
) -> Optional[Tuple[str, str]]:

    # ---------------------------------
    # Format 1:
    # {
    #     "prompt": ...,
    #     "response": ...
    # }
    # ---------------------------------

    prompt = item.get("prompt")
    response = item.get("response")

    if (
        isinstance(prompt, str)
        and isinstance(response, str)
    ):
        return prompt, response

    # ---------------------------------
    # Format 2:
    # {
    #     "request": {
    #         "prompt": ...
    #     },
    #     "response": {
    #         "generated_text": ...
    #     }
    # }
    # ---------------------------------

    request = item.get("request")
    response_obj = item.get("response")

    if (
        isinstance(request, dict)
        and isinstance(response_obj, dict)
    ):
        prompt = request.get("prompt")
        response = response_obj.get(
            "generated_text"
        )

        if (
            isinstance(prompt, str)
            and isinstance(response, str)
        ):
            return prompt, response

    return None

def load_training_pairs(
        responses_file: str,
        max_examples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:

    if not os.path.exists(responses_file):
        raise FileNotFoundError(
            f"Responses file not found: "
            f"{responses_file}"
        )

    prompts = []
    responses = []

    skipped = 0

    with open(
        responses_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            pair = extract_prompt_response(
                item
            )

            if pair is None:
                skipped += 1
                continue

            prompt, response = pair

            prompts.append(prompt)
            responses.append(response)

            if (
                max_examples is not None
                and len(prompts) >= max_examples
            ):
                break

    if len(prompts) == 0:
        raise RuntimeError(
            "No valid prompt-response pairs "
            "were found."
        )

    logger.info(
        f"Loaded {len(prompts)} "
        f"prompt-response pairs"
    )

    if skipped > 0:
        logger.warning(
            f"Skipped {skipped} invalid entries"
        )

    return prompts, responses

def train(args) -> None:

    prompts, responses = load_training_pairs(
        responses_file=args.responses_file,
        max_examples=args.max_examples,
    )

    logger.info(
        f"Reward model: "
        f"{args.reward_model_path}"
    )

    logger.info(
        f"Predictor layers: "
        f"{args.predictor_layers}"
    )

    logger.info(
        f"n_a: {args.n_a}"
    )

    logger.info(
        f"epoch_th: {args.epoch_th}"
    )

    # ---------------------------------
    # Build model
    # ---------------------------------

    model = RewardValueModel(
        reward_model_path=args.reward_model_path,
        predictor_layers=args.predictor_layers,
        device=args.device,
        exact_architecture=args.exact_architecture,
        embedding_strategy=args.embedding_strategy,
        use_projection=not args.no_projection,
        n_a=args.n_a,
        epoch_th=args.epoch_th,
        seed=args.seed,
    )

    # ---------------------------------
    # Train predictor
    # ---------------------------------

    model.train(
        prompts=prompts,
        responses=responses,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_path=args.output_path,
    )

    logger.info(
        "Training completed successfully."
    )

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train a NeuBoots-based predictor "
            "to regress reward-model scalar values."
        )
    )

    # ---------------------------------
    # Data
    # ---------------------------------

    parser.add_argument(
        "--responses-file",
        type=str,
        required=True,
        help=(
            "JSONL file containing "
            "prompt-response pairs."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Maximum number of training "
            "examples to load."
        ),
    )

    # ---------------------------------
    # Reward model
    # ---------------------------------

    parser.add_argument(
        "--reward-model-path",
        type=str,
        required=True,
    )

    # ---------------------------------
    # Predictor architecture
    # ---------------------------------

    parser.add_argument(
        "--predictor-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--exact-architecture",
        action="store_true",
    )

    parser.add_argument(
        "--embedding-strategy",
        type=str,
        default="shared_trainable",
        choices=[
            "shared_trainable",
            "shared_frozen",
            "separate",
        ],
    )

    parser.add_argument(
        "--no-projection",
        action="store_true",
        help="Disable the optional H -> H projection.",
    )

    # ---------------------------------
    # NeuBoots
    # ---------------------------------

    parser.add_argument(
        "--n-a",
        type=int,
        default=400,
        help="Number of NeuBoots training groups.",
    )

    parser.add_argument(
        "--epoch-th",
        type=int,
        default=0,
        help=(
            "Epoch threshold after which "
            "bootstrap alpha is sampled."
        ),
    )

    # ---------------------------------
    # Optimization
    # ---------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
    )

    # ---------------------------------
    # Runtime
    # ---------------------------------

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

    set_seed(args.seed)

    os.makedirs(
        args.output_path,
        exist_ok=True,
    )

    train(args)


if __name__ == "__main__":
    main()

