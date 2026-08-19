import os
import gc
import json
import argparse
import logging
import random

import numpy as np
import torch
from tqdm import tqdm

from baseline.pessimism.datasets.gsm8k import GSM8KDataset
from baseline.pessimism.models.openai_model import (
    run_openai_inference,
)
from baseline.pessimism.models.rnd_reward_model import (
    RNDRewardModel,
)

from ensemble.evaluation.score_gsm8k_neuboots import (
    load_predictor,
    score_batch,
)


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                "uuid": request_uuid,
                "prompt": problem.prompt,
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
        not isinstance(prompt, str)
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
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            pair = (
                extract_prompt_response(
                    item
                )
            )

            if pair is None:
                continue

            prompt, response = pair

            prompts.append(prompt)
            responses.append(response)

    if not prompts:
        raise RuntimeError(
            "No calibration "
            "prompt-response pairs found."
        )

    logger.info(
        f"Loaded {len(prompts)} "
        f"calibration pairs"
    )

    return (
        prompts,
        responses,
    )


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
        target_layers=config[
            "target_layers"
        ],
        predictor_layers=config[
            "predictor_layers"
        ],
        rnd_weight=config.get(
            "rnd_weight",
            0.2,
        ),
        device=device,
        exact_architecture=config.get(
            "exact_architecture",
            False,
        ),
        embedding_strategy=config.get(
            "embedding_strategy",
            "shared_trainable",
        ),
        use_projection=config.get(
            "use_projection",
            True,
        ),
    )

    model.load_predictor(
        rnd_model_path
    )

    reward_scores = []
    rnd_uncertainties = []

    for prompt, response in tqdm(
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

        # Caution returns:
        #
        # rnd_score = - MSE
        #
        # We want positive uncertainty.
        rnd_uncertainty = (
            -rnd_score
        )

        reward_scores.append(
            reward_score
        )

        rnd_uncertainties.append(
            rnd_uncertainty
        )

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        reward_scores,
        rnd_uncertainties,
    )


def score_neuboots(
        prompts,
        responses,
        checkpoint_dir,
        num_mc,
        batch_size,
        device,
):
    tokenizer, predictor = (
        load_predictor(
            checkpoint_dir=(
                checkpoint_dir
            ),
            device=device,
        )
    )

    uncertainties = []

    for start in tqdm(
        range(
            0,
            len(prompts),
            batch_size,
        ),
        desc="NeuBoots calibration scoring",
    ):
        end = min(
            start + batch_size,
            len(prompts),
        )

        batch_prompts = (
            prompts[start:end]
        )

        batch_responses = (
            responses[start:end]
        )

        # score_batch() currently expects
        # one shared prompt for the batch,
        # so calibration requires individual
        # prompt-response pairs.
        for prompt, response in zip(
            batch_prompts,
            batch_responses,
        ):
            _, mc_std = score_batch(
                tokenizer=tokenizer,
                predictor=predictor,
                prompt=prompt,
                responses=[response],
                num_mc=num_mc,
                device=device,
            )

            uncertainties.append(
                float(mc_std[0])
            )

    del predictor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return uncertainties


def save_calibration_scores(
        output_path,
        prompts,
        responses,
        reward_scores,
        rnd_uncertainties,
        neuboots_uncertainties,
):
    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for i in range(
            len(prompts)
        ):
            item = {
                "calibration_index": i,
                "prompt": prompts[i],
                "response": responses[i],
                "reward_score": float(
                    reward_scores[i]
                ),
                "rnd_uncertainty": float(
                    rnd_uncertainties[i]
                ),
                "neuboots_uncertainty": float(
                    neuboots_uncertainties[i]
                ),
            }

            f.write(
                json.dumps(item)
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()

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
        required=True,
    )

    parser.add_argument(
        "--generation-output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--responses-file",
        type=str,
        default=None,
    )

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

    set_seed(
        args.seed
    )

    # --------------------------------------------------
    # 1. Generate independent responses
    # --------------------------------------------------

    responses_file = (
        args.responses_file
    )

    if responses_file is None:

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

    # --------------------------------------------------
    # 2. Load calibration pairs
    # --------------------------------------------------

    (
        prompts,
        responses,
    ) = load_pairs(
        responses_file
    )

    # --------------------------------------------------
    # 3. Original RM + RND
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 4. NeuBoots
    # --------------------------------------------------

    neuboots_uncertainties = (
        score_neuboots(
            prompts=prompts,
            responses=responses,
            checkpoint_dir=(
                args.neuboots_checkpoint_dir
            ),
            num_mc=args.num_mc,
            batch_size=args.batch_size,
            device=args.device,
        )
    )

    # --------------------------------------------------
    # 5. Save
    # --------------------------------------------------

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
        neuboots_uncertainties=(
            neuboots_uncertainties
        ),
    )

    print()
    print(
        f"Saved calibration scores: "
        f"{args.output_path}"
    )

    print(
        f"Number of calibration pairs: "
        f"{len(prompts)}"
    )


if __name__ == "__main__":
    main()