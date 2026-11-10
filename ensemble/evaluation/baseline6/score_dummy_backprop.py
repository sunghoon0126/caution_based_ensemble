"""Stream full candidate groups and adapt each candidate to the fixed group minimum."""

import argparse
import json
from pathlib import Path

from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.reward_model_adapter import RewardModelAdapter


def run(args):
    if args.input_path.resolve() == args.output_path.resolve():
        raise ValueError("Input and output must differ")
    adapter = None
    config = dict(dummy_mode="fixed_min", learning_rate=finite(args.learning_rate),
                  num_sgd_steps=args.num_sgd_steps, optimizer="sgd", momentum=0, weight_decay=0,
                  loss="half_mse", update_scope=args.update_scope, device=args.device,
                  dummy_calibration_pool=args.num_samples, reward_model=args.reward_model,
                  backend=args.reward_backend, dtype=args.dtype, revision=args.revision,
                  max_length=args.max_length, before_rtol=args.before_rtol, before_atol=args.before_atol)
    if args.learning_rate <= 0 or args.num_sgd_steps < 0 or args.before_rtol < 0 or args.before_atol < 0:
        raise ValueError("Invalid SGD/tolerance settings")
    writer = ResumeWriter(args.output_path, config)
    expected_identity = None
    total = 0
    try:
        for group in groups(args.input_path, args.num_samples):
            dummy = min(finite(row["reward_model_score"]) for row in group)
            context = dict(dummy_reward=dummy, pool_hash=digest(group))
            for row in group:
                identity = (row.get("reward_model_path"), row.get("reward_scoring_config"), row.get("reward_model_revision"))
                if expected_identity is not None and identity != expected_identity:
                    raise ValueError("Mixed original RM scoring settings")
                expected_identity = identity
                if identity[0] != args.reward_model or not isinstance(identity[1], dict):
                    raise ValueError("Original scores require matching model and scoring metadata")
                completed, provenance = writer.completed(row, context)
                total += 1
                if completed:
                    continue
                if adapter is None:
                    adapter = RewardModelAdapter(args.reward_model, args.device, args.dtype,
                                                 args.reward_backend, args.revision, args.max_length)
                if identity[1] != adapter.settings:
                    raise ValueError("Original and BP tokenization/dtype/revision settings must match")
                resolved = getattr(adapter.model.config, "_commit_hash", None)
                if identity[2] and identity[2] != resolved:
                    raise ValueError("Resolved RM checkpoint revision differs")
                try:
                    result = adapter.adapted_score(row["prompt"], row["response"], dummy,
                                                   args.learning_rate, args.num_sgd_steps, args.update_scope)
                    import math
                    if not math.isclose(result["reward_before"], row["reward_model_score"],
                                        rel_tol=args.before_rtol, abs_tol=args.before_atol):
                        raise ValueError("Original vs BP reward_before mismatch; check batching/model/settings")
                except Exception as error:
                    failure = dict(candidate_id=row["candidate_id"], finite=False, error=repr(error),
                                   dummy_reward=dummy, **provenance)
                    with args.output_path.with_suffix(".failure.json").open("w") as dest:
                        json.dump(failure, dest, indent=2, allow_nan=False)
                    raise
                writer.write(dict(row, **result, **provenance, dummy_reward=dummy,
                                  dummy_calibration_pool=args.num_samples, finite=True,
                                  learning_rate=args.learning_rate, num_sgd_steps=args.num_sgd_steps,
                                  update_scope=args.update_scope))
        writer.finish()
        if not total:
            raise ValueError("No candidates")
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-sgd-steps", type=int, default=1)
    parser.add_argument("--dummy-mode", choices=["fixed_min"], default="fixed_min")
    parser.add_argument("--optimizer", choices=["sgd"], default="sgd")
    parser.add_argument("--momentum", type=float, choices=[0.0], default=0)
    parser.add_argument("--weight-decay", type=float, choices=[0.0], default=0)
    parser.add_argument("--loss", choices=["half_mse"], default="half_mse")
    parser.add_argument("--update-scope", choices=["full", "head_only"], default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32")
    parser.add_argument("--reward-backend", choices=["auto", "pair", "skywork"], default="auto")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--before-rtol", type=float, default=1e-4)
    parser.add_argument("--before-atol", type=float, default=1e-4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
