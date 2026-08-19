import argparse
import json
import os

import torch

from ensemble.models.neuboots_reward_model import (
    RewardValueModel,
)


def load_example(
        responses_file: str,
        index: int = 0,
):
    with open(
        responses_file,
        "r",
        encoding="utf-8",
    ) as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if index < 0 or index >= len(lines):
        raise IndexError(
            f"index={index}, "
            f"but file has {len(lines)} examples"
        )

    item = json.loads(lines[index])

    request = item["request"]
    response_obj = item["response"]

    prompt = request["prompt"]
    response = response_obj["generated_text"]

    return prompt, response


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--responses-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-mc",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--pessimism-weight",
        type=float,
        default=1.0,
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

    config_path = os.path.join(
        args.model_dir,
        "reward_predictor_config.json",
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    model = RewardValueModel(
        reward_model_path=config[
            "reward_model_path"
        ],
        predictor_layers=config[
            "predictor_layers"
        ],
        device=args.device,
        exact_architecture=config[
            "exact_architecture"
        ],
        embedding_strategy=config[
            "embedding_strategy"
        ],
        use_projection=config[
            "use_projection"
        ],
    )

    model.load_predictor(
        args.model_dir
    )

    prompt, response = load_example(
        responses_file=args.responses_file,
        index=args.index,
    )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    target_reward = model.compute_reward_score(
        prompt=prompt,
        response=response,
    )

    reward_samples = model.predict_mc(
        prompt=prompt,
        response=response,
        num_mc=args.num_mc,
    )

    reward_mean = reward_samples.mean().item()

    absolute_error = abs(
        reward_mean - target_reward
    )

    uncertainty_std = (
        model.compute_uncertainty_from_samples(
            reward_samples,
            uncertainty_type="std",
        )
    )

    uncertainty_mean_distance = (
        model.compute_uncertainty_from_samples(
            reward_samples,
            uncertainty_type="mean_distance",
        )
    )

    pessimistic_score = (
            target_reward
            - args.pessimism_weight
            * uncertainty_std
    )

    print()
    print("===== Reward comparison =====")
    print(
        f"Target RM reward     : "
        f"{target_reward:.6f}"
    )
    print(
        f"Predictor MC mean    : "
        f"{reward_mean:.6f}"
    )
    print(
        f"Absolute error       : "
        f"{absolute_error:.6f}"
    )

    print()
    print("===== NeuBoots MC =====")
    print(
        f"Num MC               : "
        f"{args.num_mc}"
    )
    print(
        f"MC samples shape     : "
        f"{tuple(reward_samples.shape)}"
    )
    print(
        f"MC samples           : "
        f"{reward_samples.tolist()}"
    )
    print(
        f"MC mean              : "
        f"{reward_mean:.6f}"
    )
    print(
        f"MC std               : "
        f"{uncertainty_std:.6f}"
    )
    print(
        f"MC mean distance     : "
        f"{uncertainty_mean_distance:.6f}"
    )

    print()
    print("===== Pessimistic score =====")
    print(
        f"Reward               : "
        f"{target_reward:.6f}"
    )
    print(
        f"Lambda               : "
        f"{args.pessimism_weight:.6f}"
    )
    print(
        f"Uncertainty          : "
        f"{uncertainty_std:.6f}"
    )
    print(
        f"Pessimistic score    : "
        f"{pessimistic_score:.6f}"
    )


if __name__ == "__main__":
    main()