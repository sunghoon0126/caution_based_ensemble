# LogiQA 2.0 / ARC-Challenge and scalar reward models

## Modules and source formats

- `baseline/pessimism/datasets/choice_answers.py`: source label normalization and strict final-line answer extraction.
- `baseline/pessimism/datasets/reasoning_choice.py`: existing `SpecialCompletionDataset`/`SpecialCompletionProblem` adapters.
- `ensemble/models/scalar_reward_inputs.py`: DeBERTa pair and Skywork chat serialization.
- `ensemble/evaluation/score_reference_candidates.py`: batched scalar inference, model selection, separate resumable output.
- `ensemble/evaluation/prepare_reference_selection.py`: train calibration and validation lambda selection.
- `ensemble/evaluation/analyze_reference_nll.py`: five methods, accuracy/length curves, summary and oracle prefix Pass@N.

LogiQA uses the official English MRC JSONL files from
[csitfun/LogiQA2.0](https://github.com/csitfun/LogiQA2.0),
`logiqa/DATA/LOGIQA/{train,dev,test}.txt`.
Public CLI splits are `train`, `validation`, `test`; validation maps to `dev.txt`.
`answer` is a zero-based integer. Use `--dataset-data-dir` for a local copy of
the three files, and `--dataset-revision COMMIT` to pin the remote source.
The dataset license is CC BY-NC-SA 4.0; consult its official repository.

ARC uses [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc),
configuration `ARC-Challenge`, with its native train/validation/test splits.
The answerKey is matched against the source labels before remapping options
in source order to A, B, C, etc. This also supports numeric source labels.
Both adapters retain the source ID, label mapping, split, revision and original
index in `problem_extra`. Invalid data raises an error rather than being skipped.

Each prompt includes the actual choices and requests `#### B` as the last line.
The checker accepts only a single uppercase available choice on that line;
bare letters in reasoning, multiple answers and trailing explanations fail.
Candidate schema and generator logprob/NLL definitions are unchanged.
GSM8K defaults to 500 generated tokens; all other supported datasets use 1024.

## Generation on the server

Run from the repository root in the existing CUDA environment. The model alias
in the inference JSON must match `/v1/models` on the server. Check that the vLLM
context window accommodates prompt tokens plus 1024 output tokens.

Small generation smoke run (execute on the server):

```bash
python -m ensemble.generation.generate_reference_candidates \
  --dataset logiqa2 --split test \
  --inference-config configs/llama32_3b_generation_with_logprobs.json \
  --output-dir outputs/smoke/logiqa2 --num-samples 2 --limit-problems 2
```

Repeat with `--dataset arc_challenge` and a separate output directory.
The following generates independent train/calibration (128 problems), validation
(128) and test (up to 1000) pools, each with 512 candidates:

```bash
bash scripts/run_reference_choice_generation.sh logiqa2
INFERENCE_CONFIG=configs/llama32_3b_generation_gpu1.json \
  bash scripts/run_reference_choice_generation.sh arc_challenge
```

Environment overrides: `CALIBRATION_PROBLEMS`, `VALIDATION_PROBLEMS`,
`TEST_PROBLEMS`, `NUM_SAMPLES`, `REFERENCE_GENERATION_ROOT`, `DATASET_REVISION`,
`DATASET_DATA_DIR`. Limits greater than split size select the whole split.
`--shuffle-problems --seed 2` shuffles the new datasets before start-index/limit
slicing. For sharding use the same revision, shuffle setting and seed in every
part, with different `--start-index` values.
Resume checks cached messages and generation settings; changed prompts or
token limits require a fresh output directory.

## Score the identical candidates with each RM

Official model cards:
[Qwen3 1.7B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B),
[Llama 3.2 3B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.2-3B).
Both use `AutoModelForSequenceClassification` and a scalar logit. The official
Skywork template accepts user/assistant messages without a system role. Here
the original candidate prompt is the user message and the unchanged response
is the assistant message. A duplicate template BOS is removed before tokenizing,
as in the official example. No generation prompt is appended. Batching uses
right padding and attention masks. No FlashAttention installation is required.

The existing requirements pin Transformers 4.57.6, which includes Qwen3 support.
The implementation uses the installed model's default attention implementation.
For server validation, first add `--max-candidates 4 --batch-size 2` and inspect
the scores. Use a separate smoke output directory.

After generation, make GPU memory available for RM inference (including any
generator vLLM processes). These commands score each native split into separate
RM directories; they do not regenerate responses:

```bash
DATASET=logiqa2  # or arc_challenge
ROOT="outputs/reference_choice/${DATASET}/trial1"
for SPLIT in train validation test; do
  CUDA_VISIBLE_DEVICES=0 python -m ensemble.evaluation.score_reference_candidates \
    --input-path "$ROOT/$SPLIT/generated_candidates.jsonl" \
    --output-path "$ROOT/$SPLIT/deberta/scored_candidates.jsonl" \
    --reward-model OpenAssistant/reward-model-deberta-v3-large-v2 \
    --device cuda:0 --batch-size 16 --compact

  CUDA_VISIBLE_DEVICES=1 python -m ensemble.evaluation.score_reference_candidates \
    --input-path "$ROOT/$SPLIT/generated_candidates.jsonl" \
    --output-path "$ROOT/$SPLIT/skywork_qwen3_1p7b/scored_candidates.jsonl" \
    --reward-model Skywork/Skywork-Reward-V2-Qwen3-1.7B \
    --device cuda:0 --dtype bfloat16 --batch-size 8 --compact
done
```

To use the secondary RM, select
`--reward-model Skywork/Skywork-Reward-V2-Llama-3.2-3B` and a separate output
directory. Auto-detection handles these exact HF IDs; local Skywork checkpoint
paths need `--reward-backend skywork`. Pin both model and tokenizer via
`--revision COMMIT`. Skywork defaults to a 16384-token input limit and rejects
overlong inputs; `--max-length` changes this limit explicitly. DeBERTa retains
its baseline truncating pair tokenizer. Reduce batch size if GPU memory runs out.

Scored records preserve candidate ID, response hash, generator length, correctness,
and both NLLs. They add `reward_model_path`, `reward_model_score`,
`reward_scoring_config` and resolved model revision. Resume rejects settings or
model changes. Old scored files without the new settings metadata should be
kept as baseline artifacts and rescored into a fresh path when using this code.

## Held-out selection and test analysis

For each RM independently, calibrate on train, choose lambdas on validation,
then evaluate the frozen configuration on test. For example:

```bash
RM=skywork_qwen3_1p7b  # repeat with deberta
RESULT="$ROOT/analysis/$RM"
python -m ensemble.evaluation.prepare_reference_selection calibrate \
  --input-paths "$ROOT/train/$RM/scored_candidates.jsonl" \
  --output-path "$RESULT/calibration.json"

python -m ensemble.evaluation.prepare_reference_selection tune \
  --input-paths "$ROOT/validation/$RM/scored_candidates.jsonl" \
  --normalization-stats "$RESULT/calibration.json" \
  --expected-candidates 512 --lambdas 0,0.01,0.05,0.1,0.2,0.5,1,2,5 \
  --output-path "$RESULT/selection.json"

python -m ensemble.evaluation.analyze_reference_nll \
  --input-paths "$ROOT/test/$RM/scored_candidates.jsonl" \
  --normalization-stats "$RESULT/calibration.json" \
  --selection-config "$RESULT/selection.json" \
  --expected-candidates 512 --output-dir "$RESULT/test"
```

Calibration uses population standard deviation over candidates. Lambda selection
maximizes validation accuracy at the full candidate count; ties keep the first
lambda in the supplied grid. The five-method analyzer supports normalized lambdas
without its historical raw-to-normalized conversion via `--lambda-space normalized`.
`--selection-config` sets this automatically. Train/test data are rejected by
validation tuning; analysis checks dataset and RM settings against calibration
and selection provenance. Keep calibration/validation/test source revisions fixed.
Stats and selection JSONs use exclusive creation so finalized choices are not
overwritten accidentally.

Outputs under `analysis/$RM/test/` include `accuracy_curves.csv`,
`selected_length_curves.csv`, `length_distribution.csv`, `summary.csv`,
`oracle_pass_at_n.csv`, their plots and `analysis_metadata.json`.
Oracle Pass@N here means the fraction of problems with at least one correct
candidate in the same first-N pool used by every selection method.
Install matplotlib in the server environment if absent from its existing setup.

## Validation scope and existing BBH caveat

Local checks use small in-memory/mock fixtures only. No dataset downloads,
model weight downloads, actual checkpoint loading, CUDA execution or full
experiments were performed locally. Run both server smoke tests before launching
512-candidate jobs; real checkpoint/API operation remains a server check.
The prior BBH prompt and subset/checker issues still make old BBH outputs
unsuitable for final performance conclusions. This integration does not repair
those historical outputs.
