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
from baseline.pessimism.datasets.gsm8k import GSM8KDataset
from baseline.pessimism.models.openai_model import run_openai_inference

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

def load_gsm8k_train_data(
        seed: int = 42,
        max_examples: Optional[int] = None,
) -> GSM8KDataset:

    logger.info(
        f"Loading GSM8K train split "
        f"(seed={seed}, max_examples={max_examples})"
    )

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

    if (
        max_examples is not None
        and max_examples < len(dataset)
    ):
        dataset.problems = dataset.problems[
            :max_examples
        ]

    logger.info(
        f"Using {len(dataset)} GSM8K "
        f"training problems"
    )

    return dataset

def generate_responses(
        dataset: GSM8KDataset,
        inference_config: Dict,
        output_path: str,
        num_samples_per_problem: int = 1,
) -> str:

    os.makedirs(
        output_path,
        exist_ok=True,
    )

    logger.info(
        f"Generating responses for "
        f"{len(dataset)} problems "
        f"({num_samples_per_problem} per problem)"
    )

    requests = []

    for problem_idx, problem in enumerate(dataset):

        prompt = problem.prompt

        if not prompt:
            logger.warning(
                f"Skipping problem {problem_idx}: "
                f"empty prompt"
            )
            continue

        for sample_idx in range(
            num_samples_per_problem
        ):

            request_uuid = (
                f"gsm8k_train_"
                f"{problem_idx}_"
                f"sample_{sample_idx}"
            )

            requests.append(
                {
                    "uuid": request_uuid,
                    "prompt": prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                }
            )

    logger.info(
        f"Prepared {len(requests)} "
        f"generation requests"
    )

    inference_type = inference_config.get(
        "type",
        "openai",
    )

    if inference_type != "openai":
        raise ValueError(
            f"Only OpenAI-compatible inference "
            f"is currently supported, got "
            f"{inference_type}"
        )

    inference_kwargs = (
        inference_config
        .get("inference_kwargs", {})
        .copy()
    )

    inference_kwargs["output_path"] = (
        output_path
    )

    run_openai_inference(
        requests=requests,
        **inference_kwargs,
    )

    responses_file = os.path.join(
        output_path,
        "all_responses.jsonl",
    )

    if not os.path.exists(responses_file):
        raise FileNotFoundError(
            f"Generated responses file "
            f"not found: {responses_file}"
        )

    logger.info(
        f"Responses saved to "
        f"{responses_file}"
    )

    return responses_file

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
        max_examples=None,
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
        default=None,
        help=(
            "Existing JSONL file containing "
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

    parser.add_argument(
        "--generate-responses",
        action="store_true",
        help=(
            "Generate GSM8K training responses "
            "before training."
        ),
    )

    parser.add_argument(
        "--inference-config",
        type=str,
        default=None,
        help=(
            "JSON configs for OpenAI-compatible "
            "LLM inference."
        ),
    )

    parser.add_argument(
        "--generation-output-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help=(
            "Number of generated responses "
            "per GSM8K problem."
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

    if args.generate_responses:

        if args.inference_config is None:
            raise ValueError(
                "--inference-configs is required "
                "when --generate-responses is used."
            )

        if args.generation_output_path is None:
            args.generation_output_path = os.path.join(
                args.output_path,
                "generated_responses",
            )

        dataset = load_gsm8k_train_data(
            seed=args.seed,
            max_examples=args.max_examples,
        )

        with open(
                args.inference_config,
                "r",
                encoding="utf-8",
        ) as f:
            inference_config = json.load(f)

        args.responses_file = generate_responses(
            dataset=dataset,
            inference_config=inference_config,
            output_path=args.generation_output_path,
            num_samples_per_problem=args.num_samples,
        )

    else:

        if args.responses_file is None:
            raise ValueError(
                "Provide --responses-file or use "
                "--generate-responses."
            )

    train(args)


if __name__ == "__main__":
    main()

