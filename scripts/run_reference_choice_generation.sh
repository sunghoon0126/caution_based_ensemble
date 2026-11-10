#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=".:${PYTHONPATH:-}"
DATASET="${1:?Usage: bash scripts/run_reference_choice_generation.sh logiqa2|arc_challenge}"
case "$DATASET" in logiqa2|arc_challenge) ;; *) echo "Unsupported dataset: $DATASET" >&2; exit 1 ;; esac
CONFIG="${INFERENCE_CONFIG:-configs/llama32_3b_generation_with_logprobs.json}"
ROOT="${REFERENCE_GENERATION_ROOT:-outputs/reference_choice}/${DATASET}/trial1"
SAMPLES="${NUM_SAMPLES:-512}"
EXTRA=()
if [[ -n "${DATASET_DATA_DIR:-}" ]]; then
  EXTRA+=(--dataset-data-dir "$DATASET_DATA_DIR")
fi
for SPLIT in train validation test; do
  case "$SPLIT" in
    train) LIMIT="${CALIBRATION_PROBLEMS:-128}" ;;
    validation) LIMIT="${VALIDATION_PROBLEMS:-128}" ;;
    test) LIMIT="${TEST_PROBLEMS:-1000}" ;;
  esac
  python -m ensemble.generation.generate_reference_candidates \
    --dataset "$DATASET" --split "$SPLIT" --trial-id 1 \
    --inference-config "$CONFIG" --output-dir "$ROOT/$SPLIT" \
    --num-samples "$SAMPLES" --limit-problems "$LIMIT" \
    --shuffle-problems --seed 2 --sample-seed-base 100000 \
    --dataset-revision "${DATASET_REVISION:-main}" "${EXTRA[@]}"
done
