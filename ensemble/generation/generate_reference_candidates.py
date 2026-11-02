#!/usr/bin/env python3

"""Generate candidates and persist their generator likelihood metadata."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from baseline.pessimism.models.generation_logprobs import (
    summarize_choice_logprobs,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def build_prompt_config() -> Dict[str, Any]:
    return {
        "system_prompt": (
            "Solve the following problem step by step. "
            "Give your final numerical answer at the end with: "
            "#### {NUM}"
        ),
        "add_generation_prompt": True,
    }


def load_dataset(
    dataset_name: str,
    split: str,
    seed: int,
):
    from baseline.pessimism.datasets.bbh import BBHDataset
    from baseline.pessimism.datasets.gsm8k import GSM8KDataset
    from baseline.pessimism.datasets.mathgsm import MathGSMDataset

    common = {
        "seed": seed,
        "split": split,
        "fewshot_num": 0,
        "limit_problems": None,
        "prompt_postprocessor_config": build_prompt_config(),
    }

    if dataset_name == "gsm8k":
        return GSM8KDataset(
            name_or_path="gsm8k",
            config_name="main",
            **common,
        )

    if dataset_name == "math500":
        return MathGSMDataset(
            math_split="math_500_test",
            **common,
        )

    if dataset_name == "bbh":
        return BBHDataset(
            shuffle_before_limit=False,
            **common,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def iter_request_chunks(
    problems,
    dataset_name: str,
    split: str,
    trial_id: int,
    num_samples: int,
    chunk_size: int,
    generation_config: Dict[str, Any],
    sample_seed_base: Optional[int],
) -> Iterable[List[Dict[str, Any]]]:
    chunk: List[Dict[str, Any]] = []

    for dataset_index, problem in problems:
        if not problem.prompt:
            raise ValueError(
                f"Problem {problem.uuid} has an empty prompt."
            )

        for sample_index in range(num_samples):
            candidate_id = (
                f"{dataset_name}_{split}_trial_{trial_id}_"
                f"{problem.uuid}_"
                f"sample_{sample_index}"
            )
            request_generation_config = generation_config.copy()
            if sample_seed_base is not None:
                request_generation_config["seed"] = (
                    sample_seed_base
                    + dataset_index * num_samples
                    + sample_index
                )
            chunk.append(
                {
                    "uuid": candidate_id,
                    "candidate_id": candidate_id,
                    "dataset": dataset_name,
                    "split": split,
                    "trial_id": trial_id,
                    "dataset_index": dataset_index,
                    "instance_id": str(problem.uuid),
                    "sample_index": sample_index,
                    "prompt": problem.prompt,
                    "problem": problem.problem,
                    "ground_truth": problem.answer,
                    "problem_extra": problem.extra,
                    "problem_generation_config": (
                        problem.generation_config
                    ),
                    "generation_config": request_generation_config,
                    "messages": [
                        {
                            "role": "user",
                            "content": problem.prompt,
                        }
                    ],
                }
            )

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if chunk:
        yield chunk


def canonicalize_record(
    raw_record: Dict[str, Any],
    problem_by_id,
    include_token_logprobs: bool = False,
) -> Dict[str, Any]:
    request = raw_record["request"]
    response = raw_record["response"]
    choices = response.get("choices")

    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(
            f"{request['candidate_id']}: expected exactly one choice, "
            f"got {0 if not isinstance(choices, list) else len(choices)}"
        )

    choice = choices[0]
    message = choice.get("message", {})
    response_text = message.get("content")
    if not isinstance(response_text, str):
        raise ValueError(
            f"{request['candidate_id']}: missing response text"
        )

    instance_id = str(request["instance_id"])
    problem = problem_by_id.get(instance_id)
    if problem is None:
        raise ValueError(
            f"{request['candidate_id']}: unknown instance_id {instance_id}"
        )

    likelihood = summarize_choice_logprobs(choice)
    if not include_token_logprobs:
        likelihood.pop("token_logprobs")
    response_digest = hashlib.sha256(
        response_text.encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": request["candidate_id"],
        "dataset": request["dataset"],
        "split": request["split"],
        "trial_id": int(request["trial_id"]),
        "dataset_index": int(request["dataset_index"]),
        "instance_id": instance_id,
        "sample_index": int(request["sample_index"]),
        "model": response.get("model"),
        "created": response.get("created"),
        "system_fingerprint": response.get("system_fingerprint"),
        "prompt": request["prompt"],
        "messages": request["messages"],
        "problem": request["problem"],
        "ground_truth": request["ground_truth"],
        "problem_extra": request.get("problem_extra", {}),
        "problem_generation_config": request.get(
            "problem_generation_config", {}
        ),
        "generation_config": request["generation_config"],
        "response": response_text,
        "response_sha256": response_digest,
        "correct": bool(problem.check_correctness(response_text)),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        **likelihood,
    }


def write_canonical_records(
    responses_dir: Path,
    output_path: Path,
    problem_by_id,
    requests: Iterable[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    count = 0
    with temporary_path.open("w", encoding="utf-8") as f:
        for request in requests:
            response_path = responses_dir / f"{request['uuid']}.json"
            if not response_path.exists():
                raise FileNotFoundError(response_path)

            with response_path.open("r", encoding="utf-8") as raw_file:
                raw_record = json.load(raw_file)
            stored_id = raw_record.get("request", {}).get("candidate_id")
            if stored_id != request["candidate_id"]:
                raise ValueError(
                    f"{response_path}: expected candidate_id "
                    f"{request['candidate_id']}, got {stored_id}"
                )

            canonical = canonicalize_record(
                raw_record,
                problem_by_id,
                include_token_logprobs=False,
            )
            f.write(json.dumps(canonical, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise RuntimeError("No generated candidate records were found")
    os.replace(temporary_path, output_path)


def select_problems(
    dataset,
    start_index: int,
    limit_problems: int,
) -> List[Tuple[int, Any]]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")

    stop_index = len(dataset)
    if limit_problems is not None:
        if limit_problems <= 0:
            raise ValueError("--limit-problems must be positive")
        stop_index = min(stop_index, start_index + limit_problems)

    if start_index >= stop_index:
        raise ValueError(
            f"Empty dataset slice [{start_index}:{stop_index}]"
        )

    return list(
        enumerate(
            dataset.problems[start_index:stop_index],
            start=start_index,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm8k", "math500", "bbh"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-problems", type=int, default=None)
    parser.add_argument("--request-chunk-size", type=int, default=10000)
    parser.add_argument("--sample-seed-base", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    from baseline.pessimism.models.openai_model import (
        run_openai_inference,
    )

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.trial_id <= 0:
        raise ValueError("--trial-id must be positive")
    if args.request_chunk_size <= 0:
        raise ValueError("--request-chunk-size must be positive")

    with args.inference_config.open("r", encoding="utf-8") as f:
        inference_config = json.load(f)
    if inference_config.get("type", "openai") != "openai":
        raise ValueError("Only OpenAI-compatible inference is supported")

    inference_kwargs = inference_config.get(
        "inference_kwargs", {}
    ).copy()
    generation_config = inference_kwargs.get(
        "generation_config", {}
    ).copy()
    if generation_config.get("logprobs") is not True:
        raise ValueError(
            "generation_config.logprobs must be true"
        )
    if int(generation_config.get("n", 1)) != 1:
        raise ValueError(
            "generation_config.n must be 1; use --num-samples "
            "for repeated candidates"
        )
    sample_seed_base = args.sample_seed_base
    if sample_seed_base is None and "seed" in generation_config:
        sample_seed_base = int(generation_config.pop("seed"))
        logger.info(
            "Using generation_config.seed=%d as the per-candidate "
            "seed base",
            sample_seed_base,
        )
    inference_kwargs["generation_config"] = generation_config
    inference_kwargs["output_path"] = str(args.output_dir)
    inference_kwargs["dump_individual_rsp"] = True
    inference_kwargs["write_aggregate_rsp"] = False

    dataset = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        seed=args.seed,
    )
    selected = select_problems(
        dataset=dataset,
        start_index=args.start_index,
        limit_problems=args.limit_problems,
    )
    expected_count = len(selected) * args.num_samples
    logger.info(
        "Generating %d candidates for %d problems",
        expected_count,
        len(selected),
    )

    first_chunk = True
    for chunk_index, requests in enumerate(
        iter_request_chunks(
            problems=selected,
            dataset_name=args.dataset,
            split=args.split,
            trial_id=args.trial_id,
            num_samples=args.num_samples,
            chunk_size=args.request_chunk_size,
            generation_config=generation_config,
            sample_seed_base=sample_seed_base,
        ),
        start=1,
    ):
        logger.info(
            "Running request chunk %d (%d requests)",
            chunk_index,
            len(requests),
        )
        run_openai_inference(
            requests=requests,
            reset_output_file=first_chunk,
            **inference_kwargs,
        )
        first_chunk = False

    canonical_path = args.output_dir / "generated_candidates.jsonl"
    problem_by_id = {
        str(problem.uuid): problem
        for _, problem in selected
    }
    write_canonical_records(
        responses_dir=args.output_dir / "responses",
        output_path=canonical_path,
        problem_by_id=problem_by_id,
        requests=(
            request
            for chunk in iter_request_chunks(
                problems=selected,
                dataset_name=args.dataset,
                split=args.split,
                trial_id=args.trial_id,
                num_samples=args.num_samples,
                chunk_size=args.request_chunk_size,
                generation_config=generation_config,
                sample_seed_base=sample_seed_base,
            )
            for request in chunk
        ),
    )
    logger.info("Saved canonical candidates: %s", canonical_path)


if __name__ == "__main__":
    main()
