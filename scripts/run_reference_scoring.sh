
#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-gsm8k}"
TRIAL="${2:-1}"

ROOT="outputs/reference_generation/${DATASET}/trial${TRIAL}"
MODEL="OpenAssistant/reward-model-deberta-v3-large-v2"

mkdir -p "${ROOT}/logs"

echo "Scoring ${DATASET} trial${TRIAL}"

# GPU 0: part0
(
  CUDA_VISIBLE_DEVICES=0 \
  python -m ensemble.evaluation.score_reference_candidates \
    --input-path "${ROOT}/part0/generated_candidates.jsonl" \
    --output-path "${ROOT}/part0/scored_candidates.jsonl" \
    --reward-model "${MODEL}" \
    --device cuda:0 \
    --batch-size 16 \
    --log-every 1000 \
    --compact \
    2>&1 | tee "${ROOT}/logs/scoring_gpu0.log"
) &
PID0=$!

# GPU 1: part1
(
  CUDA_VISIBLE_DEVICES=1 \
  python -m ensemble.evaluation.score_reference_candidates \
    --input-path "${ROOT}/part1/generated_candidates.jsonl" \
    --output-path "${ROOT}/part1/scored_candidates.jsonl" \
    --reward-model "${MODEL}" \
    --device cuda:0 \
    --batch-size 16 \
    --log-every 1000 \
    --compact \
    2>&1 | tee "${ROOT}/logs/scoring_gpu1.log"
) &
PID1=$!

STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "Scoring failed. GPU0=${STATUS0}, GPU1=${STATUS1}"
  exit 1
fi

# Verify that all generated candidates were scored.
for PART in part0 part1; do
  INPUT="${ROOT}/${PART}/generated_candidates.jsonl"
  SCORED="${ROOT}/${PART}/scored_candidates.jsonl"

  EXPECTED=$(wc -l < "${INPUT}")
  ACTUAL=$(wc -l < "${SCORED}")

  if [[ "${EXPECTED}" -ne "${ACTUAL}" ]]; then
    echo "${PART}: expected ${EXPECTED}, got ${ACTUAL}"
    exit 1
  fi
done

# Merge in original generation order.
TMP="${ROOT}/scored_candidates.jsonl.tmp"
OUTPUT="${ROOT}/scored_candidates.jsonl"

cat \
  "${ROOT}/part0/scored_candidates.jsonl" \
  "${ROOT}/part1/scored_candidates.jsonl" \
  > "${TMP}"

mv "${TMP}" "${OUTPUT}"

echo "Scoring complete"
wc -l "${OUTPUT}"
