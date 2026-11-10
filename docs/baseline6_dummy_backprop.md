# Baseline 6: Fixed-pool dummy reward backpropagation

Branch: `exp/baseline6-dummy-backprop`, based on reference-likelihood commit
`847f2f7`. All implementation lives under `ensemble/evaluation/baseline6/`;
the existing reference-likelihood experiment modules are reused without edits.

## Algorithm

For each complete problem group of N_max candidates, first save original scores
`r_i = f_theta0(x,y_i)`. Set `dummy = min_i r_i` over **all N_max** candidates.
For each candidate separately, start at theta0, compute
`loss = 0.5 * (f_theta(x,y_i) - dummy)**2`, perform SGD, then forward the same
candidate with the updated parameters. The returned `reward_after` is the
selection score. There is no persistent training, momentum, weight decay or
cross-candidate optimizer state. `torch.func.functional_call` receives detached
parameters and private cloned buffers; updates create fresh tensors. Original
parameter gradients are not populated. Multiple steps detach between steps,
so this performs ordinary SGD rather than meta-gradient optimization.

Main defaults: full update, 1 step, half-MSE, SGD without momentum/weight decay,
float32. Learning rate is required explicitly and has no claimed optimum.
`head_only` additionally supports DeBERTa-v2 pooler+classifier and Qwen3/Llama
score heads. `last_block_and_head`, LR sweeps, gradient-penalty approximation
and other optimizers are future extensions, not implemented modes.

Selection uses the original first N candidates with earliest-candidate tie
breaking. Even at N=1 or N=4, BP uses the full N_max dummy pool. Thus this is
fixed-pool test-time adaptation, not a compute-matched ordinary BoN comparison.
Reward reduction does not establish reduced prediction error or accuracy gains.
No probability guarantee from `Pessimism_Proof.pdf` is claimed.

## Files and outputs

- `generate_candidates.py`: reuse dataset adapters, request chunking, API client;
  accepts an existing OpenAI inference config and optional logprobs.
- `score_original_rm.py`: shared scalar scorer plus stricter full-source resume
  verification; no `--compact` because BP requires prompt/response.
- `reward_model_adapter.py`: model-specific serialization, stateless actual SGD.
- `score_dummy_backprop.py`: fixed group minimum, bounded group streaming,
  per-candidate diagnostic records, config/source/pool fingerprint resume.
- `common.py`: JSONL readers, SQLite disk-backed duplicate-ID detection, resume.
- `evaluate_bon.py`: compare original/BP paired trials; average trial accuracies
  at each N before finding the peak. Reject changed pools/settings/nonfinite data.
- `plot_bon.py`: PNG/PDF curve rendering.
- `smoke_test.py`: explicitly invoked server real-checkpoint checks.
- `run.py` / `scripts/run_baseline6.sh`: independent-stage orchestration.
- `configs/baseline6/experiment.json`: sanity settings reference, not an executable
  experiment configuration. The runner uses CLI arguments and saves its exact
  manifest to output `experiment.json`.

Output structure:

```text
<output>/
  experiment.json
  responses/<candidate_id>.json       # if generating new data
  generated_candidates.jsonl          # if generating new data
  original_rm_scores.jsonl            # full source text + RM scores/metadata
  backprop_scores.jsonl
  backprop_scores.failure.json         # only on failed candidate
  evaluation/
    curves.csv
    summary.csv
    evaluation_metadata.json
    accuracy_vs_n.png
    accuracy_vs_n.pdf
```

BP rows preserve source data and add dummy_reward, dummy_calibration_pool,
reward_before/after/change, gradient_norm (first step), gradient_norms (each step),
dummy_loss_before/after, learning_rate, num_sgd_steps, update_scope,
backprop_config and input_fingerprint. A group fingerprint binds all original
scores, so changing a future candidate invalidates prior rows of that group.
Each successful row is flushed; restart validates completed prefix without
recomputing BP. Failure metadata is written separately and the process stops;
failed scores never enter evaluation. A truncated final JSONL row must be
inspected/repaired explicitly. No rows are silently dropped or overwritten.

## Server setup and sanity sequence

Use the existing PyTorch/Transformers CUDA environment. PyTorch 2.x with
torch.func is required. Plotting requires matplotlib. Baseline 6 does not
start a vLLM server; generation calls an already running OpenAI-compatible API.
The generator CLI argument must match the actual server alias, not necessarily
the Hugging Face checkpoint name. Make GPU memory available before RM BP.
Full-model gradients plus updated tensors require considerably more memory
than ordinary scoring. All real checkpoint/CUDA steps below are unexecuted
locally; run them explicitly on the server.

1. Run actual autograd and pipeline unit tests:

```bash
python -m unittest discover -s tests -p 'test_baseline6.py' -v
```

2. Generate the 10-problem / 16-candidate sanity pool (default seed 42):

```bash
python -m ensemble.evaluation.baseline6.generate_candidates \
  --generator-model llama32-3b-instruct --dataset gsm8k \
  --generation-config configs/llama32_3b_generation_with_logprobs.json \
  --output-dir outputs/baseline6/sanity_pool \
  --limit-problems 10 --num-samples 16
```

Logprobs are optional: use a copy of the inference JSON with `logprobs` omitted
or false if desired. Existing configs enable them and retain the original NLLs.
The generation JSON controls temperature. Token limits default to GSM8K 500,
other datasets 1024; `--max-tokens` overrides these defaults. This wrapper does
not change existing generation code.

3. Choose a small pilot LR explicitly; the following numeric value is illustrative,
not optimized. Run actual checkpoint restoration/order/k=0/minimum-loss checks:

```bash
python -m ensemble.evaluation.baseline6.smoke_test \
  --generated-input-path outputs/baseline6/sanity_pool/generated_candidates.jsonl \
  --reward-model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --num-samples 16 --learning-rate 1e-6 --device cuda:0 \
  --output-path outputs/baseline6/sanity_smoke.json
```

4. Reuse that exact pool for the end-to-end sanity comparison:

```bash
bash scripts/run_baseline6.sh \
  --generator-model llama32-3b-instruct --dataset gsm8k \
  --reward-model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --generated-input-path outputs/baseline6/sanity_pool/generated_candidates.jsonl \
  --output-dir outputs/baseline6/sanity_lr1e-6 \
  --num-samples 16 --learning-rate 1e-6 --num-sgd-steps 1 \
  --device cuda:0 --dtype float32
```

With existing inputs, prepare complete selected groups first; runner slicing
flags apply only to new generation. Required dataset/split/trial/generator
identity is checked. No generation API is needed for pool reuse.

For new generation through the runner, omit `--generated-input-path`, provide
`--generation-config`, and set `--limit-problems 10 --num-samples 16`.
The default candidate count is 512; specify 16 explicitly for sanity runs.
Do not launch full runs before measuring runtime and peak GPU memory.

## Independent stages and stronger RMs

```bash
python -m ensemble.evaluation.baseline6.score_original_rm \
  --input-path outputs/baseline6/sanity_pool/generated_candidates.jsonl \
  --output-path outputs/baseline6/independent/original_rm_scores.jsonl \
  --reward-model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --device cuda:0 --dtype float32 --batch-size 1

python -m ensemble.evaluation.baseline6.score_dummy_backprop \
  --input-path outputs/baseline6/independent/original_rm_scores.jsonl \
  --output-path outputs/baseline6/independent/backprop_scores.jsonl \
  --reward-model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --device cuda:0 --dtype float32 --num-samples 16 --learning-rate 1e-6

python -m ensemble.evaluation.baseline6.evaluate_bon \
  --original-input-paths outputs/baseline6/independent/original_rm_scores.jsonl \
  --backprop-input-paths outputs/baseline6/independent/backprop_scores.jsonl \
  --num-samples 16 --output-dir outputs/baseline6/independent/evaluation
```

Original scoring batch size defaults to 1 in the runner to match candidatewise
BP reward_before. BP rejects before-score disagreements beyond 1e-4 absolute
or relative tolerance. Investigate differences rather than blindly increasing
tolerances. RM backend/dtype/revision/max_length must agree in both stages.
Existing canonical reference candidate JSONL is accepted without regeneration.
Compact scored-only inputs cannot be used for BP because text is missing.

Skywork Qwen3 1.7B / Llama 3.2 3B use the shared chat backend; local checkpoint
paths require `--reward-backend skywork`. Pin `--revision` for both stages.
Actual gradient execution on these checkpoints remains a server validation task.

For multiple trials, provide paired ordered lists to `--original-input-paths`
and `--backprop-input-paths`. Same ordered problem set and SGD configuration
are required; trial IDs must differ.

After sanity passes, explicitly create a 100-problem × 64-candidate pilot pool.
Choose LR on validation/pilot data, recording runtime, reward/ranking changes
and accuracy. Only then consider 1000 × 512; no full experiment is auto-launched.
No generated data or checkpoints should be committed.

## Local verification limits

Local pipeline/JSONL/resume/evaluation tests run without downloads. PyTorch is
not installed locally, so the four actual autograd tests are skipped here and
must pass on the server. No CUDA experiments, weight downloads or accuracy
claims have been made. Existing reference-likelihood tests remain available.

PyTorch API reference:
https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html
