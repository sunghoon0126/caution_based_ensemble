"""Compare first-N selection with a fixed full-pool dummy, averaging trials first."""

import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path

from ensemble.evaluation.baseline6.common import digest, finite, groups


def evaluate(original_paths, backprop_paths, count):
    if len(original_paths) != len(backprop_paths):
        raise ValueError("Original and BP path lists must pair by trial")
    n_values = [n for n in range(1, count + 1) if n & (n - 1) == 0]
    if not n_values or n_values[-1] != count:
        raise ValueError("Candidate count must be a positive power of two")
    trials, identity, settings = [], None, None
    instance_pool = None
    for original_path, bp_path in zip(original_paths, backprop_paths):
        totals = {method: dict.fromkeys(n_values, 0) for method in ("original_rm", "dummy_backprop_rm")}
        num_problems, trial_id, pool = 0, None, []
        for original, adapted in zip_longest(groups(original_path, count), groups(bp_path, count)):
            if original is None or adapted is None:
                raise ValueError("Original/BP candidate pools differ")
            dummy = min(finite(r["reward_model_score"]) for r in original)
            context = dict(dummy_reward=dummy, pool_hash=digest(original))
            pool.append((original[0]["dataset_index"], original[0]["instance_id"]))
            num_problems += 1
            oracle_best = {method: float("-inf") for method in totals}
            correct = dict.fromkeys(totals, 0)
            for n, (src, bp) in enumerate(zip(original, adapted), 1):
                if bp.get("candidate_id") != src["candidate_id"] or bp.get("input_fingerprint") != digest({"source": src, "context": context}):
                    raise ValueError("Candidate identity/source fingerprint mismatch")
                if bp.get("dummy_reward") != dummy or bp.get("dummy_calibration_pool") != count:
                    raise ValueError("Not the fixed full-pool minimum")
                if bp.get("finite") is not True:
                    raise ValueError("Non-finite/failed BP candidate")
                current_identity = (src["dataset"], src["split"], src["reward_model_path"], src.get("reward_scoring_config"), src.get("reward_model_revision"), src.get("model"))
                if identity is not None and identity != current_identity:
                    raise ValueError("Mixed experiment/RM settings")
                identity = current_identity
                # Device may differ between independent trials; other SGD settings must match.
                current_settings = {k: v for k, v in bp["backprop_config"].items() if k != "device"}
                if settings is not None and settings != current_settings:
                    raise ValueError("Mixed BP settings")
                settings = current_settings
                trial_id = src["trial_id"]
                truth = src["correct"]
                if type(truth) not in (bool, int) or truth not in (0, 1):
                    raise ValueError("Invalid correctness")
                scores = dict(original_rm=finite(src["reward_model_score"]), dummy_backprop_rm=finite(bp["reward_after"]))
                for method, score in scores.items():
                    if score > oracle_best[method]:
                        oracle_best[method], correct[method] = score, int(truth)
                    if n in totals[method]:
                        totals[method][n] += correct[method]
        if not num_problems:
            raise ValueError("Empty trial")
        if any(t["trial_id"] == trial_id for t in trials):
            raise ValueError("Duplicate trial")
        if instance_pool is not None and pool != instance_pool:
            raise ValueError("Trials must use the same ordered problem set")
        instance_pool = pool
        trials.append(dict(trial_id=trial_id, num_problems=num_problems, totals=totals))
    curves, summary = [], []
    for method in ("original_rm", "dummy_backprop_rm"):
        points = []
        for n in n_values:
            accuracy = 100 * sum(t["totals"][method][n] / t["num_problems"] for t in trials) / len(trials)
            row = dict(method=method, N=n, accuracy_percent=accuracy, num_trials=len(trials), dummy_calibration_pool=count)
            curves.append(row)
            points.append(row)
        peak = max(points, key=lambda r: r["accuracy_percent"])
        summary.append(dict(method=method, peak_n=peak["N"], peak_accuracy_percent=peak["accuracy_percent"],
                            final_n=count, final_accuracy_percent=points[-1]["accuracy_percent"],
                            degradation_pp=peak["accuracy_percent"] - points[-1]["accuracy_percent"],
                            dummy_calibration_pool=count))
    for row in curves:
        base = next(r["accuracy_percent"] for r in curves if r["method"] == "original_rm" and r["N"] == row["N"])
        row["difference_vs_original_pp"] = row["accuracy_percent"] - base
    for row in summary:
        row["final_difference_vs_original_pp"] = row["final_accuracy_percent"] - summary[0]["final_accuracy_percent"]
    return curves, summary, dict(dataset=identity[0], split=identity[1], reward_model=identity[2],
                                 backprop_config=settings, trial_ids=[t["trial_id"] for t in trials],
                                 num_problems_per_trial=[t["num_problems"] for t in trials],
                                 dummy_calibration_pool=count, selection_budgets=n_values,
                                 interpretation="Fixed full-pool dummy test-time adaptation; N is selection budget only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-input-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--backprop-input-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    curves, summary, metadata = evaluate(args.original_input_paths, args.backprop_input_paths, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("curves.csv", curves), ("summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="") as dest:
            writer = csv.DictWriter(dest, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    with (args.output_dir / "evaluation_metadata.json").open("w") as dest:
        json.dump(metadata, dest, indent=2)
    if not args.skip_plots:
        from ensemble.evaluation.baseline6.plot_bon import plot
        plot(args.output_dir / "curves.csv", args.output_dir)


if __name__ == "__main__":
    main()
