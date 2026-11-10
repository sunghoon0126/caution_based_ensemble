"""Explicit server-only real-checkpoint sanity test; never launched automatically."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from ensemble.evaluation.baseline6.common import groups
from ensemble.evaluation.baseline6.reward_model_adapter import RewardModelAdapter


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-input-path", type=Path, required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    candidates = next(groups(args.generated_input_path, args.num_samples))
    if len(candidates) < 2:
        raise ValueError("At least two candidates required")
    adapter = RewardModelAdapter(args.reward_model, args.device, "float32")
    before_hash = state_digest(adapter.model)
    rewards = [adapter.score(r["prompt"], r["response"]) for r in candidates]
    dummy = min(rewards)
    minimum = candidates[rewards.index(dummy)]
    unchanged = adapter.adapted_score(minimum["prompt"], minimum["response"], dummy, args.learning_rate)
    if unchanged["gradient_norm"] != 0 or unchanged["reward_after"] != dummy:
        raise AssertionError("Zero-loss minimum candidate changed")
    def adapt(index, steps=1):
        r = candidates[index]
        return adapter.adapted_score(r["prompt"], r["response"], dummy, args.learning_rate, steps)
    a, b = adapt(0), adapt(1)
    reverse_b, reverse_a = adapt(1), adapt(0)
    for forward, reverse in ((a, reverse_a), (b, reverse_b)):
        if not math.isclose(forward["reward_after"], reverse["reward_after"], rel_tol=1e-5, abs_tol=1e-5):
            raise AssertionError("Candidate order dependence")
    zero = adapt(0, 0)
    if zero["reward_before"] != zero["reward_after"]:
        raise AssertionError("k=0 changed reward")
    if not math.isclose(adapter.score(candidates[1]["prompt"], candidates[1]["response"]), rewards[1], rel_tol=1e-5, abs_tol=1e-5):
        raise AssertionError("Original score changed")
    if state_digest(adapter.model) != before_hash:
        raise AssertionError("Original model state changed")
    import torch
    result = dict(dummy_reward=dummy, dummy_calibration_pool=len(candidates), original_state_sha256=before_hash,
                  checks_passed=True, candidates=[a, b], minimum_candidate=unchanged,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
                  reward_moved_down=[r["reward_after"] <= r["reward_before"] for r in (a, b)])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x") as dest:
        json.dump(result, dest, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
