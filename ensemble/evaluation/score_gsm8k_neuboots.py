import os
import json
import argparse
import logging
import random

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer

from ensemble.models.neuboots_reward_model import RewardValuePredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    reward_model_path = config["reward_model_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        reward_model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    predictor = RewardValuePredictor(
        model_path=reward_model_path,
        num_layers=config["predictor_layers"],
        exact_architecture=config["exact_architecture"],
        embedding_strategy=config["embedding_strategy"],
        use_projection=config["use_projection"],
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    predictor.load_state_dict(state_dict)
    predictor.eval()

    logger.info(
        f"Loaded NeuBoots predictor from {checkpoint_path}"
    )

    return tokenizer, predictor


@torch.inference_mode()
def score_batch(
        tokenizer,
        predictor,
        prompt: str,
        responses,
        num_mc: int,
        device: str,
):
    prompts = [prompt] * len(responses)

    inputs = tokenizer(
        prompts,
        responses,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    reward_samples = predictor(
        input_ids,
        attention_mask,
        alpha=num_mc,
    )

    # reward_samples:
    # [num_mc, batch_size]

    if reward_samples.dim() != 2:
        raise RuntimeError(
            "Expected reward_samples shape "
            f"[num_mc, batch_size], got "
            f"{tuple(reward_samples.shape)}"
        )

    mc_mean = reward_samples.mean(
        dim=0
    )

    mc_std = reward_samples.std(
        dim=0,
        unbiased=False,
    )

    return (
        mc_mean.cpu().tolist(),
        mc_std.cpu().tolist(),
    )


def load_completed_ids(
        output_path: str,
):
    completed = set()

    if not os.path.exists(output_path):
        return completed

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            completed.add(
                item["instance_id"]
            )

    return completed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--caution-results",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

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

    set_seed(args.seed)

    output_dir = os.path.dirname(
        args.output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    tokenizer, predictor = load_predictor(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    completed_ids = load_completed_ids(
        args.output_path
    )

    if completed_ids:
        logger.info(
            f"Found {len(completed_ids)} "
            f"already scored problems"
        )

    processed = 0

    with open(
        args.caution_results,
        "r",
        encoding="utf-8",
    ) as input_file:

        for line in tqdm(
            input_file,
            desc="Scoring GSM8K problems",
        ):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            instance_id = item["instance_id"]

            if instance_id in completed_ids:
                continue

            if (
                args.max_problems is not None
                and processed >= args.max_problems
            ):
                break

            prompt = item["prompt"]
            responses = item["all_samples"]

            if len(responses) != 512:
                logger.warning(
                    f"{instance_id}: expected 512 "
                    f"responses, got {len(responses)}"
                )

            all_mc_mean = []
            all_uncertainty = []

            for start in range(
                0,
                len(responses),
                args.batch_size,
            ):
                end = min(
                    start + args.batch_size,
                    len(responses),
                )

                batch_responses = responses[
                    start:end
                ]

                mc_mean, mc_std = score_batch(
                    tokenizer=tokenizer,
                    predictor=predictor,
                    prompt=prompt,
                    responses=batch_responses,
                    num_mc=args.num_mc,
                    device=args.device,
                )

                all_mc_mean.extend(
                    mc_mean
                )

                all_uncertainty.extend(
                    mc_std
                )

            result = {
                "instance_id": instance_id,
                "num_candidates": len(responses),
                "num_mc": args.num_mc,
                "all_neuboots_mc_mean": all_mc_mean,
                "all_neuboots_uncertainty": all_uncertainty,
            }

            with open(
                args.output_path,
                "a",
                encoding="utf-8",
            ) as output_file:
                output_file.write(
                    json.dumps(result)
                    + "\n"
                )

            processed += 1

    logger.info(
        f"Finished scoring {processed} problems"
    )


if __name__ == "__main__":
    main()