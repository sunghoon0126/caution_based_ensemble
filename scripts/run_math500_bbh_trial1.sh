
#!/usr/bin/env bash
set -euo pipefail

cd ~/workplace/caution_based_ensemble
export PYTHONPATH=.

CONFIG0="configs/llama32_3b_generation_with_logprobs.json"
CONFIG1="configs/llama32_3b_generation_gpu1.json"

ROOT="${REFERENCE_GENERATION_ROOT:-outputs/reference_generation_promptfix_1024}"
LOG_DIR="logs/reference_generation"
MAX_TOKENS="${MAX_TOKENS:-1024}"

mkdir -p "$LOG_DIR"

# Check both inference servers.
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null

echo "Starting MATH-500 and BBH generation"

# GPU 0: MATH-500 Trial 1
(
  python -m ensemble.generation.generate_reference_candidates \
    --dataset math500 \
    --split test \
    --trial-id 1 \
    --inference-config "$CONFIG0" \
    --output-dir "$ROOT/math500/trial1" \
    --num-samples 512 \
    --start-index 0 \
    --limit-problems 500 \
    --request-chunk-size 10000 \
    --sample-seed-base 100000 \
    --seed 2 \
    --max-tokens "$MAX_TOKENS" \
    2>&1 | tee "$LOG_DIR/math500_trial1.log"
) &
PID0=$!

# GPU 1: BBH Trial 1
(
  python -m ensemble.generation.generate_reference_candidates \
    --dataset bbh \
    --split test \
    --trial-id 1 \
    --inference-config "$CONFIG1" \
    --output-dir "$ROOT/bbh/trial1" \
    --num-samples 512 \
    --start-index 0 \
    --limit-problems 1000 \
    --request-chunk-size 10000 \
    --sample-seed-base 100000 \
    --seed 2 \
    --max-tokens "$MAX_TOKENS" \
    2>&1 | tee "$LOG_DIR/bbh_trial1.log"
) &
PID1=$!

STATUS0=0
STATUS1=0

wait "$PID0" || STATUS0=$?
wait "$PID1" || STATUS1=$?

if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "Generation failed: math500=$STATUS0, bbh=$STATUS1"
  exit 1
fi

echo "Generation completed"

wc -l "$ROOT/math500/trial1/generated_candidates.jsonl"
wc -l "$ROOT/bbh/trial1/generated_candidates.jsonl"
