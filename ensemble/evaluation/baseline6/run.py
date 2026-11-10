"""Explicit stage orchestration. No experiment runs until this CLI is invoked."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from ensemble.evaluation.baseline6.common import groups


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
    parser.add_argument("--update-scope", choices=["full", "head_only"], default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--reward-backend", default="auto", choices=["auto", "pair", "skywork"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Original RM batch size; 1 minimizes before-score differences")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.num_sgd_steps < 0:
        parser.error("Positive finite learning-rate and nonnegative num-sgd-steps required")
    if args.num_samples <= 0 or args.num_samples & (args.num_samples - 1):
        parser.error("num-samples must be a positive power of two for evaluation")
    if args.batch_size <= 0 or args.trial_id <= 0:
        parser.error("batch-size/trial-id must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def call(module, flags):
        subprocess.run([sys.executable, "-m", "ensemble.evaluation.baseline6." + module] + list(map(str, flags)), check=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest_path = args.output_dir / "experiment.json"
    if manifest_path.exists():
        with manifest_path.open() as source:
            if json.load(source) != manifest:
                raise ValueError("Experiment settings differ; use a new output-dir")
    else:
        with manifest_path.open("x") as dest:
            json.dump(manifest, dest, indent=2)
    generated = args.generated_input_path
    if generated is None:
        if args.generation_config is None:
            parser.error("New generation requires --generation-config")
        flags = ["--generator-model", args.generator_model, "--dataset", args.dataset,
                 "--generation-config", args.generation_config, "--output-dir", args.output_dir,
                 "--split", args.split, "--trial-id", args.trial_id, "--num-samples", args.num_samples,
                 "--start-index", args.start_index, "--seed", args.seed]
        if args.limit_problems is not None:
            flags += ["--limit-problems", args.limit_problems]
        if args.max_tokens is not None:
            flags += ["--max-tokens", args.max_tokens]
        call("generate_candidates", flags)
        generated = args.output_dir / "generated_candidates.jsonl"
    elif args.start_index != 0 or args.limit_problems is not None:
        parser.error("Existing input must already contain the selected complete problem groups; slicing flags apply to new generation only")
    found = False
    for group in groups(generated, args.num_samples):
        found = True
        for row in group:
            if (row["dataset"], row["split"], row["trial_id"]) != (args.dataset, args.split, args.trial_id):
                raise ValueError("Generated input dataset/split/trial mismatch")
            if row.get("requested_generator_model", row.get("model")) != args.generator_model:
                raise ValueError("Generated input model differs from --generator-model (use recorded server alias)")
    if not found:
        raise ValueError("Empty generated input")
    original = args.output_dir / "original_rm_scores.jsonl"
    adapted = args.output_dir / "backprop_scores.jsonl"
    flags = ["--reward-model", args.reward_model, "--device", args.device, "--dtype", args.dtype,
             "--reward-backend", args.reward_backend, "--revision", args.revision]
    if args.max_length:
        flags += ["--max-length", args.max_length]
    call("score_original_rm", ["--input-path", generated, "--output-path", original,
                               "--batch-size", args.batch_size] + flags)
    call("score_dummy_backprop", ["--input-path", original, "--output-path", adapted,
                                  "--num-samples", args.num_samples, "--learning-rate", args.learning_rate,
                                  "--num-sgd-steps", args.num_sgd_steps, "--update-scope", args.update_scope] + flags)
    call("evaluate_bon", ["--original-input-paths", original, "--backprop-input-paths", adapted,
                          "--num-samples", args.num_samples, "--output-dir", args.output_dir / "evaluation"]
         + (["--skip-plots"] if args.skip_plots else []))


if __name__ == "__main__":
    main()
