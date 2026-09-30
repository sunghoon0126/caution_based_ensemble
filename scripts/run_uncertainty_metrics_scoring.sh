#!/usr/bin/env bash

set -euo pipefail

GPU=1
NUM_MC=20
BATCH_SIZE=32
SEED=42

DATASETS=(gsm8k math500 bbh)
TRIALS=(1 2 3)

PROJECT_ROOT="$HOME/workplace/caution_based_ensemble"

CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/uncertainty_metrics/gsm8k_lightweight_s42/checkpoints/epoch_005"

SCORE_ROOT="${PROJECT_ROOT}/outputs/uncertainty_metrics/scored_5metrics"

LOG_ROOT="${PROJECT_ROOT}/logs/uncertainty_metrics"

mkdir -p \
    "$SCORE_ROOT" \
    "$LOG_ROOT"

cd "$PROJECT_ROOT"

export PYTHONPATH=.

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_FILE="${LOG_ROOT}/score_5metrics_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1


echo
echo "============================================================"
echo "NeuBoots 5-metric scoring"
echo "============================================================"
echo "GPU        : ${GPU}"
echo "Checkpoint : ${CHECKPOINT_DIR}"
echo "MC samples : ${NUM_MC}"
echo "Batch size : ${BATCH_SIZE}"
echo "Datasets   : ${DATASETS[*]}"
echo "Trials     : ${TRIALS[*]}"
echo "Output     : ${SCORE_ROOT}"
echo "Log        : ${LOG_FILE}"
echo "============================================================"
echo


get_input_path() {

    local dataset="$1"
    local trial="$2"

    case "$dataset" in

        gsm8k)
            echo "${PROJECT_ROOT}/outputs/table3_repro/baseline/trial${trial}_input.jsonl"
            ;;

        math500)
            echo "${PROJECT_ROOT}/outputs/fig3_neuboots/math500/trial${trial}_input.jsonl"
            ;;

        bbh)
            echo "${PROJECT_ROOT}/outputs/fig3_neuboots/bbh/trial${trial}_input.jsonl"
            ;;

        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 1
            ;;
    esac
}


echo
echo "============================================================"
echo "PRE-FLIGHT CHECK"
echo "============================================================"


if [[ ! -f "${CHECKPOINT_DIR}/reward_predictor.pt" ]]; then
    echo "Missing predictor:"
    echo "  ${CHECKPOINT_DIR}/reward_predictor.pt"
    exit 1
fi


if [[ ! -f "${CHECKPOINT_DIR}/reward_predictor_config.json" ]]; then
    echo "Missing predictor config:"
    echo "  ${CHECKPOINT_DIR}/reward_predictor_config.json"
    exit 1
fi


for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        INPUT=$(get_input_path "$dataset" "$trial")

        if [[ ! -f "$INPUT" ]]; then
            echo
            echo "Missing scorer input:"
            echo "  dataset=${dataset}"
            echo "  trial=${trial}"
            echo "  path=${INPUT}"
            echo
            exit 1
        fi

        echo "[OK] ${dataset} trial${trial}"
        echo "     ${INPUT}"

    done

done


echo
echo "Pre-flight check passed."
echo


for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        INPUT=$(get_input_path \
            "$dataset" \
            "$trial"
        )

        OUTPUT_DIR="${SCORE_ROOT}/${dataset}"

        OUTPUT="${OUTPUT_DIR}/trial${trial}.jsonl"

        mkdir -p "$OUTPUT_DIR"

        echo
        echo "============================================================"
        echo "Scoring"
        echo "Dataset : ${dataset}"
        echo "Trial   : ${trial}"
        echo "Input   : ${INPUT}"
        echo "Output  : ${OUTPUT}"
        echo "============================================================"
        echo

        CUDA_VISIBLE_DEVICES="$GPU" \
        python -u \
            ensemble/evaluation/score_gsm8k_neuboots.py \
            --caution-results "$INPUT" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --output-path "$OUTPUT" \
            --num-mc "$NUM_MC" \
            --batch-size "$BATCH_SIZE" \
            --device cuda \
            --seed "$SEED"

        echo
        echo -n "Scored problems: "
        wc -l < "$OUTPUT"

    done

done


echo
echo "============================================================"
echo "ALL SCORING COMPLETED"
echo "============================================================"
echo

find "$SCORE_ROOT" \
    -type f \
    -name "*.jsonl" \
    -print \
    -exec wc -l {} \;

echo
echo "Log:"
echo "  ${LOG_FILE}"
echo
