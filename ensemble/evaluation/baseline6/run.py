
"""Baseline 6 experiment runner with stage-level logging."""

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ensemble.evaluation.baseline6.common import groups


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [Baseline6] {message}", flush=True)


def call(stage_name, module, flags):
    command = [
        sys.executable,
        "-u",
        "-m",
        "ensemble.evaluation.baseline6." + module,
        *map(str, flags),
    ]

    log("=" * 65)
    log(f"[START] {stage_name}")
    log(f"[COMMAND] {' '.join(command)}")

    started = time.perf_counter()

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        log(
            f"[FAILED] {stage_name} "
            f"(exit code {error.returncode})"
        )
        raise

    elapsed = time.perf_counter() - started

    log(f"[DONE] {stage_name} ({elapsed:.1f}s)")
    log("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--generated-input-path", type=Path)
    parser.add_argument("--generation-config", type=Path)

    parser.add_argument("--split", default="test")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-problems", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int)

    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-sgd-steps", type=int, default=1)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")

    # Monitoring
    parser.add_argument("--rm-log-every", type=int, default=1000)
    parser.add_argument("--bp-log-every", type=int, default=1000)
    parser.add_argument("--bp-progress-every", type=int, default=32)
    parser.add_argument("--bp-problem-log-every", type=int, default=1)
    parser.add_argument("--no-bp-progress", action="store_true")

    args = parser.parse_args()

    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or args.num_sgd_steps < 0
    ):
        parser.error(
            "Positive finite learning rate and nonnegative steps required"
        )

    if (
        args.num_samples <= 0
        or args.num_samples & (args.num_samples - 1)
    ):
        parser.error("num-samples must be a positive power of two")

    if args.batch_size <= 0 or args.trial_id <= 0:
        parser.error("batch-size/trial-id must be positive")

    if min(
        args.rm_log_every,
        args.bp_log_every,
        args.bp_progress_every,
        args.bp_problem_log_every,
    ) <= 0:
        parser.error("Progress/log intervals must be positive")

    started = time.perf_counter()

    log("Baseline 6: Fixed Dummy-Reward Backpropagation")
    log(f"Generator: {args.generator_model}")
    log(f"Reward Model: {args.reward_model}")
    log(f"Dataset: {args.dataset}, split={args.split}")
    log(f"Trial: {args.trial_id}")
    log(f"Candidates per problem: {args.num_samples}")
    log(f"Device: {args.device}")
    log(f"SGD steps: {args.num_sgd_steps}")
    log(f"LR: {args.learning_rate:g}")
    log(f"Update scope: {args.update_scope}")
    log(f"Output: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }

    manifest_path = args.output_dir / "experiment.json"

    if manifest_path.exists():
        with manifest_path.open() as source:
            if json.load(source) != manifest:
                raise ValueError(
                    "Experiment settings differ; use a new output-dir"
                )
        log("[RESUME] Existing experiment config verified")
    else:
        with manifest_path.open("x") as dest:
            json.dump(manifest, dest, indent=2)
        log("[INIT] Saved experiment config")

    generated = args.generated_input_path

    # Stage 1
    if generated is None:
        if args.generation_config is None:
            parser.error(
                "New generation requires --generation-config"
            )

        flags = [
            "--generator-model", args.generator_model,
            "--dataset", args.dataset,
            "--generation-config", args.generation_config,
            "--output-dir", args.output_dir,
            "--split", args.split,
            "--trial-id", args.trial_id,
            "--num-samples", args.num_samples,
            "--start-index", args.start_index,
            "--seed", args.seed,
        ]

        if args.limit_problems is not None:
            flags += ["--limit-problems", args.limit_problems]

        if args.max_tokens is not None:
            flags += ["--max-tokens", args.max_tokens]

        call(
            "Stage 1: Candidate Generation",
            "generate_candidates",
            flags,
        )

        generated = args.output_dir / "generated_candidates.jsonl"

    elif (
        args.start_index != 0
        or args.limit_problems is not None
    ):
        parser.error(
            "Existing input must already contain complete selected groups; "
            "slicing flags apply only to new generation"
        )
    else:
        log(f"[REUSE] Existing generation: {generated}")

    # Validate groups
    log("[VALIDATE] Checking dataset, model, candidate groups...")

    validate_started = time.perf_counter()
    total_problems = 0
    total_candidates = 0

    for group in groups(generated, args.num_samples):
        total_problems += 1

        for row in group:
            total_candidates += 1

            if (
                row["dataset"],
                row["split"],
                row["trial_id"],
            ) != (
                args.dataset,
                args.split,
                args.trial_id,
            ):
                raise ValueError(
                    "Generated input dataset/split/trial mismatch"
                )

            recorded_generator = row.get(
                "requested_generator_model",
                row.get("model"),
            )

            if recorded_generator != args.generator_model:
                raise ValueError(
                    "Generated input model differs "
                    "from --generator-model"
                )

    if not total_problems:
        raise ValueError("Empty generated input")

    log(
        f"[VALIDATE] {total_problems:,} problems, "
        f"{total_candidates:,} candidates "
        f"({time.perf_counter() - validate_started:.1f}s)"
    )

    original = args.output_dir / "original_rm_scores.jsonl"
    adapted = args.output_dir / "backprop_scores.jsonl"

    common_flags = [
        "--reward-model", args.reward_model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--reward-backend", args.reward_backend,
        "--revision", args.revision,
    ]

    if args.max_length:
        common_flags += ["--max-length", args.max_length]

    # Stage 2: Original RM
    call(
        "Stage 2: Original RM Scoring",
        "score_original_rm",
        [
            "--input-path", generated,
            "--output-path", original,
            "--batch-size", args.batch_size,
            "--log-every", args.rm_log_every,
        ] + common_flags,
    )

    # Stage 3: Dummy Backprop
    bp_flags = [
        "--input-path", original,
        "--output-path", adapted,
        "--num-samples", args.num_samples,
        "--learning-rate", args.learning_rate,
        "--num-sgd-steps", args.num_sgd_steps,
        "--update-scope", args.update_scope,
        "--log-every", args.bp_log_every,
        "--progress-every", args.bp_progress_every,
        "--problem-log-every", args.bp_problem_log_every,
    ] + common_flags

    if args.no_bp_progress:
        bp_flags += ["--no-progress"]

    call(
        "Stage 3: Fixed Dummy-Reward Backprop Scoring",
        "score_dummy_backprop",
        bp_flags,
    )

    # Stage 4: BoN evaluation and plots
    evaluation_flags = [
        "--original-input-paths", original,
        "--backprop-input-paths", adapted,
        "--num-samples", args.num_samples,
        "--output-dir", args.output_dir / "evaluation",
    ]

    if args.skip_plots:
        evaluation_flags += ["--skip-plots"]

    call(
        "Stage 4: Best-of-N Evaluation & Plotting",
        "evaluate_bon",
        evaluation_flags,
    )

    log("=" * 65)
    log("[COMPLETE] Baseline 6 experiment finished")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")
    log(f"Results: {args.output_dir / 'evaluation'}")
    log("=" * 65)


if __name__ == "__main__":
    main()
