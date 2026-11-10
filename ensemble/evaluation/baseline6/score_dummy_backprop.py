
"""Fixed-pool dummy-reward backpropagation with progress monitoring."""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from ensemble.evaluation.baseline6.common import (
    ResumeWriter,
    digest,
    finite,
    groups,
)
from ensemble.evaluation.baseline6.reward_model_adapter import (
    RewardModelAdapter,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def status(message, bar=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [Baseline6 BP] {message}"

    if bar is not None:
        bar.write(line, file=sys.stderr)
    else:
        print(line, file=sys.stderr, flush=True)


def count_jsonl_rows(path):
    """Count candidates without loading the JSONL into memory."""
    count = 0

    with Path(path).open("rb") as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            count += block.count(b"\n")

    return count


def gpu_memory_info(device):
    if not str(device).startswith("cuda"):
        return None

    try:
        import torch

        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        return {
            "allocated": allocated,
            "reserved": reserved,
            "peak": peak,
        }
    except Exception:
        return None


def run(args):
    if args.input_path.resolve() == args.output_path.resolve():
        raise ValueError("Input and output must differ")

    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or args.num_sgd_steps < 0
        or args.before_rtol < 0
        or args.before_atol < 0
    ):
        raise ValueError("Invalid SGD/tolerance settings")

    log_every = getattr(args, "log_every", 1000)
    progress_every = getattr(args, "progress_every", 32)
    problem_log_every = getattr(args, "problem_log_every", 1)
    no_progress = getattr(args, "no_progress", False)

    if min(log_every, progress_every, problem_log_every) <= 0:
        raise ValueError("Progress/log intervals must be positive")

    config = dict(
        dummy_mode="fixed_min",
        learning_rate=finite(args.learning_rate),
        num_sgd_steps=args.num_sgd_steps,
        optimizer="sgd",
        momentum=0,
        weight_decay=0,
        loss="half_mse",
        update_scope=args.update_scope,
        device=args.device,
        dummy_calibration_pool=args.num_samples,
        reward_model=args.reward_model,
        backend=args.reward_backend,
        dtype=args.dtype,
        revision=args.revision,
        max_length=args.max_length,
        before_rtol=args.before_rtol,
        before_atol=args.before_atol,
    )

    started = time.perf_counter()

    status("=" * 60)
    status("Dummy-Reward Backpropagation Scoring")
    status(f"Input: {args.input_path}")
    status(f"Output: {args.output_path}")
    status(f"Reward Model: {args.reward_model}")
    status(f"Device: {args.device}")
    status(f"Dtype: {args.dtype}")
    status(f"SGD steps: {args.num_sgd_steps}")
    status(f"Learning rate: {args.learning_rate:g}")
    status(f"Update scope: {args.update_scope}")
    status(f"Dummy mode: fixed_min")
    status("=" * 60)

    # Phase 1: Read input size
    status("[INIT] Counting input candidates...")

    scan_start = time.perf_counter()
    total_expected = count_jsonl_rows(args.input_path)

    if total_expected == 0:
        raise ValueError("No candidates")

    if total_expected % args.num_samples != 0:
        raise ValueError(
            f"Candidate count {total_expected} is not divisible "
            f"by num_samples={args.num_samples}"
        )

    total_problems = total_expected // args.num_samples

    status(
        f"[INIT] Found {total_expected:,} candidates "
        f"across {total_problems:,} problems "
        f"(scan {time.perf_counter() - scan_start:.1f}s)"
    )

    if tqdm is None and not no_progress:
        status(
            "[WARNING] tqdm is not installed. "
            "Install with: python -m pip install tqdm"
        )

    bar = None

    if tqdm is not None and not no_progress:
        bar = tqdm(
            total=total_expected,
            desc="BP scoring",
            unit="cand",
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.1,
            file=sys.stderr,
        )

    # ResumeWriter preserves original fingerprint/resume validation.
    writer = None
    adapter = None
    expected_identity = None

    total_seen = 0
    newly_scored = 0
    resumed = 0
    problems_processed = 0

    try:
        writer = ResumeWriter(args.output_path, config)

        status("[INIT] Resume validation enabled", bar)
        status("[INIT] Starting candidate groups...", bar)

        for problem_index, group in enumerate(
            groups(args.input_path, args.num_samples),
            start=1,
        ):
            problem_started = time.perf_counter()

            dummy = min(
                finite(row["reward_model_score"])
                for row in group
            )

            context = {
                "dummy_reward": dummy,
                "pool_hash": digest(group),
            }

            last_result = None

            for row in group:
                identity = (
                    row.get("reward_model_path"),
                    row.get("reward_scoring_config"),
                    row.get("reward_model_revision"),
                )

                if (
                    expected_identity is not None
                    and identity != expected_identity
                ):
                    raise ValueError(
                        "Mixed original RM scoring settings"
                    )

                expected_identity = identity

                if (
                    identity[0] != args.reward_model
                    or not isinstance(identity[1], dict)
                ):
                    raise ValueError(
                        "Original scores require matching model "
                        "and scoring metadata"
                    )

                completed, provenance = writer.completed(
                    row, context
                )

                total_seen += 1

                if completed:
                    resumed += 1

                    if bar is not None:
                        bar.update(1)

                    continue

                # Lazy model loading: only when a pending candidate exists.
                if adapter is None:
                    status(
                        "[LOAD] Loading Reward Model checkpoint...",
                        bar,
                    )

                    load_started = time.perf_counter()

                    adapter = RewardModelAdapter(
                        args.reward_model,
                        args.device,
                        args.dtype,
                        args.reward_backend,
                        args.revision,
                        args.max_length,
                    )

                    status(
                        f"[LOAD] Reward Model loaded in "
                        f"{time.perf_counter() - load_started:.1f}s",
                        bar,
                    )

                    status(
                        f"[LOAD] Resolved backend: "
                        f"{adapter.settings['backend']}",
                        bar,
                    )

                    memory = gpu_memory_info(args.device)

                    if memory is not None:
                        status(
                            "[CUDA] "
                            f"Allocated={memory['allocated']:.2f} GiB, "
                            f"Reserved={memory['reserved']:.2f} GiB",
                            bar,
                        )

                if identity[1] != adapter.settings:
                    raise ValueError(
                        "Original and BP tokenization/dtype/revision "
                        "settings must match"
                    )

                resolved = getattr(
                    adapter.model.config,
                    "_commit_hash",
                    None,
                )

                if identity[2] and identity[2] != resolved:
                    raise ValueError(
                        "Resolved RM checkpoint revision differs"
                    )

                try:
                    result = adapter.adapted_score(
                        row["prompt"],
                        row["response"],
                        dummy,
                        args.learning_rate,
                        args.num_sgd_steps,
                        args.update_scope,
                    )

                    if not math.isclose(
                        result["reward_before"],
                        row["reward_model_score"],
                        rel_tol=args.before_rtol,
                        abs_tol=args.before_atol,
                    ):
                        raise ValueError(
                            "Original vs BP reward_before mismatch; "
                            "check batching/model/settings"
                        )

                except Exception as error:
                    failure = dict(
                        candidate_id=row["candidate_id"],
                        finite=False,
                        error=repr(error),
                        dummy_reward=dummy,
                        **provenance,
                    )

                    failure_path = (
                        args.output_path.with_suffix(".failure.json")
                    )

                    with failure_path.open("w") as dest:
                        json.dump(
                            failure,
                            dest,
                            indent=2,
                            allow_nan=False,
                        )

                    status(
                        f"[ERROR] Candidate {row['candidate_id']}: "
                        f"{error!r}",
                        bar,
                    )
                    raise

                writer.write(
                    dict(
                        row,
                        **result,
                        **provenance,
                        dummy_reward=dummy,
                        dummy_calibration_pool=args.num_samples,
                        finite=True,
                        learning_rate=args.learning_rate,
                        num_sgd_steps=args.num_sgd_steps,
                        update_scope=args.update_scope,
                    )
                )

                newly_scored += 1
                last_result = result

                if bar is not None:
                    bar.update(1)

                    if newly_scored % progress_every == 0:
                        postfix = {
                            "problem": (
                                f"{problem_index}/{total_problems}"
                            ),
                            "new": newly_scored,
                            "resume": resumed,
                            "d_reward": (
                                f"{result['reward_change']:+.3f}"
                            ),
                            "grad": (
                                f"{result['gradient_norm']:.1f}"
                            ),
                        }

                        memory = gpu_memory_info(args.device)

                        if memory is not None:
                            postfix["GPU"] = (
                                f"{memory['allocated']:.1f}G"
                            )

                        bar.set_postfix(postfix, refresh=False)

                if newly_scored % log_every == 0:
                    elapsed = time.perf_counter() - started

                    status(
                        f"[PROGRESS] Seen={total_seen:,}/"
                        f"{total_expected:,} "
                        f"New={newly_scored:,} "
                        f"Resumed={resumed:,} "
                        f"Elapsed={elapsed:.1f}s",
                        bar,
                    )

            problems_processed += 1

            if (
                problem_index == 1
                or problem_index % problem_log_every == 0
                or problem_index == total_problems
            ):
                elapsed_problem = (
                    time.perf_counter() - problem_started
                )

                message = (
                    f"[PROBLEM] {problem_index:,}/"
                    f"{total_problems:,} "
                    f"dataset_index={group[0]['dataset_index']} "
                    f"dummy={dummy:.5f} "
                    f"elapsed={elapsed_problem:.1f}s"
                )

                if last_result is not None:
                    message += (
                        f" last_reward_change="
                        f"{last_result['reward_change']:+.5f}"
                    )

                status(message, bar)

        writer.finish()

        if not total_seen:
            raise ValueError("No candidates")

        elapsed = time.perf_counter() - started
        rate = newly_scored / elapsed if elapsed > 0 else 0

        status("=" * 60, bar)
        status("[COMPLETE] BP Scoring finished", bar)
        status(f"Problems: {problems_processed:,}", bar)
        status(f"Candidates: {total_seen:,}", bar)
        status(f"Newly scored: {newly_scored:,}", bar)
        status(f"Resumed: {resumed:,}", bar)
        status(f"Total elapsed: {elapsed:.1f}s", bar)
        status(f"Average new scoring rate: {rate:.2f} cand/s", bar)

        memory = gpu_memory_info(args.device)

        if memory is not None:
            status(
                f"[CUDA] Peak allocated memory: "
                f"{memory['peak']:.2f} GiB",
                bar,
            )

        status(f"[COMPLETE] Saved to {args.output_path}", bar)
        status("=" * 60, bar)

    except KeyboardInterrupt:
        status(
            "[INTERRUPTED] Stopped by user. "
            f"Completed={total_seen:,}, "
            f"new={newly_scored:,}, resumed={resumed:,}. "
            "Completed rows are preserved for resume.",
            bar,
        )
        raise

    finally:
        if writer is not None:
            writer.close()

        if bar is not None:
            bar.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--reward-model", required=True)

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-sgd-steps", type=int, default=1)

    parser.add_argument(
        "--dummy-mode",
        choices=["fixed_min"],
        default="fixed_min",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd"],
        default="sgd",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        choices=[0.0],
        default=0,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        choices=[0.0],
        default=0,
    )
    parser.add_argument(
        "--loss",
        choices=["half_mse"],
        default="half_mse",
    )

    parser.add_argument(
        "--update-scope",
        choices=["full", "head_only"],
        default="full",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16", "auto"],
        default="float32",
    )
    parser.add_argument(
        "--reward-backend",
        choices=["auto", "pair", "skywork"],
        default="auto",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int)

    parser.add_argument(
        "--before-rtol",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--before-atol",
        type=float,
        default=1e-4,
    )

    # Monitoring options (do not affect model/scoring).
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print detailed progress every N new candidates",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=32,
        help="Update progress bar metrics every N new candidates",
    )
    parser.add_argument(
        "--problem-log-every",
        type=int,
        default=1,
        help="Print problem completion every N problems",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (regular logs remain)",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
