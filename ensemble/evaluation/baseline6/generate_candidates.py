"""Reuse dataset/request construction; generation logprobs are optional for Baseline 6."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ensemble.generation.generate_reference_candidates import (
    canonicalize_record, iter_request_chunks, load_dataset, resolve_max_tokens,
    select_problems, validate_cached_requests,
)


def generate(args):
    from baseline.pessimism.models.openai_model import run_openai_inference
    with args.generation_config.open() as source:
        config = json.load(source)
    if config.get("type", "openai") != "openai":
        raise ValueError("Expected OpenAI-compatible inference config")
    kwargs = dict(config["inference_kwargs"])
    kwargs["openai_model"] = args.generator_model
    settings = dict(kwargs.get("generation_config", {}))
    settings["max_tokens"] = resolve_max_tokens(args.dataset, args.max_tokens)
    settings.setdefault("temperature", 1.0)
    if settings.get("n", 1) != 1:
        raise ValueError("Use --num-samples, not generation_config.n")
    settings.pop("seed", None)
    dataset = load_dataset(args.dataset, args.split, args.seed)
    problems = select_problems(dataset, args.start_index, args.limit_problems)
    by_id = {str(problem.uuid): problem for _, problem in problems}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses = args.output_dir / "responses"
    def chunks():
        return iter_request_chunks(problems, args.dataset, args.split, args.trial_id,
                                   args.num_samples, 1000, settings, args.seed)
    kwargs.update(output_path=str(args.output_dir), generation_config=settings,
                  dump_individual_rsp=True, write_aggregate_rsp=False)
    for chunk in chunks():
        validate_cached_requests(responses, chunk)
        for request in chunk:
            path = responses / f"{request['uuid']}.json"
            if path.exists():
                with path.open() as source:
                    prior = json.load(source)
                if prior["request"].get("requested_generator_model") != args.generator_model:
                    raise ValueError("Cached generator differs; use new output directory")
            request["requested_generator_model"] = args.generator_model
        run_openai_inference(requests=chunk, **kwargs)
    output = args.output_dir / "generated_candidates.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as dest:
        for chunk in chunks():
            for request in chunk:
                with (responses / f"{request['uuid']}.json").open() as source:
                    raw = json.load(source)
                choices = raw["response"]["choices"]
                if len(choices) != 1:
                    raise ValueError("Expected one response choice")
                choice = choices[0]
                if choice.get("logprobs"):
                    row = canonicalize_record(raw, by_id)
                else:
                    text = choice["message"]["content"]
                    if not isinstance(text, str):
                        raise ValueError("Missing response text")
                    row = {key: value for key, value in request.items() if key != "uuid"}
                    row.update(schema_version=1, response=text, response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                               model=raw["response"].get("model"), finish_reason=choice.get("finish_reason"),
                               usage=raw["response"].get("usage"),
                               correct=bool(by_id[str(request["instance_id"])].check_correctness(text)))
                row["requested_generator_model"] = args.generator_model
                dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "bbh", "logiqa2", "arc_challenge"], required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-problems", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.trial_id <= 0:
        raise ValueError("num-samples/trial-id must be positive")
    generate(args)


if __name__ == "__main__":
    main()
