"""Fit normalization on train, then select lambdas on validation only."""

import argparse
import json
import math
from pathlib import Path

from ensemble.evaluation.evaluate_reference_nll import load_problem_groups, parse_lambdas

FIELDS = {"reward": "reward_model_score", "sequence_nll": "sequence_nll", "average_nll": "average_nll"}


def inspect_inputs(paths, required_split):
    identity = None
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != required_split:
                    raise ValueError(f"Expected {required_split} data, got {row['split']}")
                current = {"dataset": row["dataset"], "reward_model_path": row["reward_model_path"],
                           "reward_model_revision": row.get("reward_model_revision"),
                           "generator_model": row.get("model"),
                           "reward_scoring_config": row.get("reward_scoring_config")}
                if identity is not None and current != identity:
                    raise ValueError("Mixed dataset or reward model/settings")
                identity = current
    if identity is None:
        raise ValueError("No scored candidates")
    return identity


def calibrate(paths):
    identity = inspect_inputs(paths, "train")
    count = 0
    means = dict.fromkeys(FIELDS, 0.0)
    m2 = dict.fromkeys(FIELDS, 0.0)
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                for name, field in FIELDS.items():
                    value = float(row[field])
                    if not math.isfinite(value):
                        raise ValueError("Non-finite calibration value")
                    delta = value - means[name]
                    means[name] += delta / count
                    m2[name] += delta * (value - means[name])
    result = {"metadata": dict(identity, split="train", count=count, input_paths=list(map(str, paths)))}
    for name in FIELDS:
        std = math.sqrt(max(0.0, m2[name] / count))
        if std <= 0:
            raise ValueError(f"Zero calibration variance: {name}")
        result[name] = {"mean": means[name], "std": std}
    return result


def tune(paths, stats, lambdas, expected_candidates):
    identity = inspect_inputs(paths, "validation")
    for key, value in identity.items():
        if stats["metadata"].get(key) != value:
            raise ValueError(f"Calibration/validation mismatch: {key}")
    totals = {name: dict.fromkeys(lambdas, 0) for name in ("sequence_nll", "average_nll")}
    count = 0
    for _, candidates in load_problem_groups(paths, expected_candidates):
        count += 1
        for name, column in (("sequence_nll", 1), ("average_nll", 2)):
            for lam in lambdas:
                def score(candidate):
                    reward = (candidate[0] - stats["reward"]["mean"]) / stats["reward"]["std"]
                    nll = (candidate[column] - stats[name]["mean"]) / stats[name]["std"]
                    return reward - lam * nll
                totals[name][lam] += max(candidates, key=score)[3]
    if not count:
        raise ValueError("No validation problems")
    chosen = {name: max(lambdas, key=totals[name].get) for name in totals}
    return dict(identity, split="validation", lambda_space="normalized",
                selection_objective=f"accuracy_at_N={expected_candidates}", num_problems=count,
                seq_lambda=chosen["sequence_nll"], avg_lambda=chosen["average_nll"],
                sweep={name: {str(lam): correct / count for lam, correct in scores.items()}
                       for name, scores in totals.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "tune"))
    parser.add_argument("--input-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--lambdas", default="0,0.01,0.05,0.1,0.2,0.5,1,2,5")
    parser.add_argument("--expected-candidates", type=int, default=512)
    args = parser.parse_args()
    if args.output_path.resolve() in [path.resolve() for path in args.input_paths]:
        raise ValueError("Output must differ from inputs")
    if args.mode == "calibrate":
        result = calibrate(args.input_paths)
    else:
        if args.normalization_stats is None:
            parser.error("tune requires --normalization-stats")
        with args.normalization_stats.open() as source:
            stats = json.load(source)
        result = tune(args.input_paths, stats, parse_lambdas(args.lambdas), args.expected_candidates)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
