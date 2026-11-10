#!/usr/bin/env python3
"""Score reference-likelihood candidates with a pair or chat scalar RM.

Input: canonical generated_candidates.jsonl (one candidate per line).
Output: scored_candidates.jsonl, preserving source order and adding RM logits.

The default DeBERTa backend preserves Caution pair tokenization. Skywork uses
its checkpoint's chat template and returns a single sequence-classifier logit.

Batching pads only within a batch and passes attention masks. Completed output
must be an exact prefix of the input, allowing safe resume after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO

from ensemble.models.scalar_reward_inputs import resolve_backend, tokenize_rewards

LOG = logging.getLogger("reference_rm_scoring")
DEFAULT_REWARD_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"
COMPACT_FIELDS = (
    "schema_version",
    "candidate_id",
    "dataset",
    "split",
    "trial_id",
    "dataset_index",
    "instance_id",
    "sample_index",
    "model",
    "response_sha256",
    "correct",
    "finish_reason",
    "token_count",
    "sequence_logprob",
    "sequence_nll",
    "average_logprob",
    "average_nll",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Caution RM rewards to generated_candidates.jsonl"
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--reward-backend", choices=("auto", "pair", "skywork"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-length", type=int, default=None, help="RM input limit; Skywork defaults to 16384 and rejects overlong inputs")
    parser.add_argument("--revision", default="main", help="Reward model/tokenizer checkpoint revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Score at most this many input candidates; useful for smoke tests")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--compact", action="store_true",
                        help="Save evaluation fields only instead of entire source record")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace prior scored output (otherwise resume)")
    return parser.parse_args()


def decode_line(raw: str, path: Path, line_number: int) -> Dict[str, Any]:
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}, line {line_number}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object in {path}, line {line_number}")
    return record


def sha_for_record(record: Dict[str, Any]) -> str:
    digest = record.get("response_sha256")
    if digest is not None:
        return str(digest)
    response = record.get("response")
    if not isinstance(response, str):
        raise ValueError("Cannot verify resume without response_sha256 or response")
    return hashlib.sha256(response.encode("utf-8")).hexdigest()


def match_completed_prefix(
    source: TextIO,
    output_path: Path,
    reward_model_name: str,
    compact: bool,
    max_candidates: Optional[int],
    input_path: Path,
    scoring_config=None,
) -> int:
    """Consume the completed output prefix and the corresponding input lines."""
    if not output_path.exists():
        return 0
    # A crash during a write may leave a truncated final JSONL row.
    # Refuse to append to that file rather than silently corrupt its contents.
    if output_path.stat().st_size:
        with output_path.open("rb") as check:
            check.seek(-1, 2)
            if check.read(1) != b"\n":
                raise ValueError(
                    f"{output_path} does not end with a newline; "
                    "inspect or repair the last record before resuming"
                )
    completed = 0
    with output_path.open("r", encoding="utf-8") as result:
        for completed, line in enumerate(result, start=1):
            if max_candidates is not None and completed > max_candidates:
                raise ValueError("Output has more rows than --max-candidates")
            src_line = source.readline()
            if not src_line:
                raise ValueError("Output contains more rows than input")
            original = decode_line(src_line, input_path, completed)
            scored = decode_line(line, output_path, completed)
            original_id = original.get("candidate_id")
            if not original_id or scored.get("candidate_id") != original_id:
                raise ValueError(
                    f"Resume order mismatch at row {completed}: "
                    f"{original_id!r} vs {scored.get('candidate_id')!r}"
                )
            if sha_for_record(original) != sha_for_record(scored):
                raise ValueError(f"Resume response hash mismatch at row {completed}")
            if scored.get("reward_model_path") != reward_model_name:
                raise ValueError(
                    f"Resume reward-model mismatch at row {completed}: "
                    f"{scored.get('reward_model_path')!r}"
                )
            if scoring_config is not None and scored.get("reward_scoring_config") != scoring_config:
                raise ValueError("Resume scoring settings differ; use a separate output path")
            if ("response" not in scored) != compact:
                raise ValueError("Resume output format differs from --compact")
            score = scored.get("reward_model_score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Invalid existing reward at row {completed}")
    return completed


def load_reward_model(model_name: str, device: str, scoring_config=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    settings = scoring_config or {"backend": "pair", "dtype": "auto", "max_length": None, "revision": "main"}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        revision=settings["revision"],
    )
    dtype = settings["dtype"]
    model_kwargs = {"num_labels": 1} if settings["backend"] == "skywork" else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=settings["revision"],
        torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        **model_kwargs,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Model tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if settings["backend"] == "skywork":
        tokenizer.padding_side = "right"
        if not tokenizer.chat_template:
            raise ValueError("Skywork tokenizer has no chat template")
    model.reference_scoring_config = settings
    if model.config.num_labels != 1:
        raise ValueError(
            f"Expected a scalar reward model (num_labels=1), got "
            f"num_labels={model.config.num_labels}"
        )
    return tokenizer, model


def score_batch(tokenizer, model, records: List[Dict[str, Any]], device: str) -> List[float]:
    import torch
    settings = model.reference_scoring_config
    tokenized = tokenize_rewards(
        tokenizer, records, settings["backend"], settings["max_length"],
    )
    tokenized = {name: tensor.to(device) for name, tensor in tokenized.items()}
    with torch.inference_mode():
        logits = model(**tokenized).logits
    values = logits.reshape(len(records), -1)
    if values.shape[1] != 1:
        raise ValueError(f"Expected one reward per candidate, got {tuple(values.shape)}")
    scores = [float(value) for value in values[:, 0].detach().cpu().tolist()]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Non-finite reward model score")
    return scores


def write_batch(
    dest: TextIO,
    records: List[Dict[str, Any]],
    tokenizer,
    model,
    device: str,
    reward_model_name: str,
    compact: bool,
) -> int:
    scores = score_batch(tokenizer, model, records, device)
    for record, score in zip(records, scores):
        row = (
            {key: record[key] for key in COMPACT_FIELDS if key in record}
            if compact else dict(record)
        )
        row["response_sha256"] = sha_for_record(record)
        row["reward_model_path"] = reward_model_name
        row["reward_scoring_config"] = model.reference_scoring_config
        row["reward_model_revision"] = getattr(model.config, "_commit_hash", None)
        row["reward_model_score"] = score
        dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    dest.flush()
    return len(records)


def run(args: argparse.Namespace) -> None:
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    scoring_config = {
        "backend": resolve_backend(args.reward_model, args.reward_backend),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "revision": args.revision,
    }
    if args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("--batch-size and --log-every must be positive")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    source_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    if source_path == output_path:
        raise ValueError("Input and output paths must differ")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    with source_path.open("r", encoding="utf-8") as source:
        already_done = match_completed_prefix(
            source,
            output_path,
            args.reward_model,
            args.compact,
            args.max_candidates,
            source_path,
            scoring_config,
        )
        LOG.info("Completed prefix: %d candidates", already_done)
        if args.max_candidates is not None and already_done >= args.max_candidates:
            return

        # Do not initialize a model when there are no pending records.
        first_pending = source.readline()
        if not first_pending:
            LOG.info("No pending candidates")
            return
        first_row = decode_line(first_pending, source_path, already_done + 1)
        if not first_row.get("candidate_id"):
            raise ValueError("Missing candidate_id")
        tokenizer, model = load_reward_model(args.reward_model, args.device, scoring_config)

        buffer = [first_row]
        processed = 0
        with output_path.open("a", encoding="utf-8") as dest:
            while True:
                if args.max_candidates is not None and already_done + processed + len(buffer) >= args.max_candidates:
                    stop = True
                else:
                    stop = False
                if not stop and len(buffer) < args.batch_size:
                    line = source.readline()
                    if line:
                        row = decode_line(
                            line, source_path, already_done + processed + len(buffer) + 1
                        )
                        if not row.get("candidate_id"):
                            raise ValueError("Missing candidate_id")
                        buffer.append(row)
                        continue
                    stop = True
                if buffer:
                    n = write_batch(
                        dest, buffer, tokenizer, model, args.device,
                        args.reward_model, args.compact
                    )
                    processed += n
                    buffer.clear()
                    if processed % args.log_every < n:
                        LOG.info("Scored %d new candidates (%d total)", processed, processed + already_done)
                if stop:
                    break
        LOG.info("Complete: %d total scored candidates -> %s", already_done + processed, output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
