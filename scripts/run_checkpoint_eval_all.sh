#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Basic settings
# ============================================================

GPU=1
NUM_MC=20
BATCH_SIZE=32
LAMBDA=0.8
NUM_BOOTSTRAP=1000
SEED=42

EPOCHS=(0 1 2 3 4 5)
DATASETS=(gsm8k math500 bbh)
TRIALS=(1 2 3)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT="$HOME/workplace/caution_based_ensemble"

CAUTION_ROOT="$HOME/workplace/Caution"

CAUTION_EVAL_ROOT="${CAUTION_ROOT}/outputs/repro_v2/repro_v2_eval_20260825_173851"

CHECKPOINT_ROOT="${PROJECT_ROOT}/outputs/checkpoint_dynamics/gsm8k_lightweight/checkpoints"

NORMALIZATION_ROOT="${PROJECT_ROOT}/outputs/checkpoint_dynamics/normalization"

SCORE_ROOT="${PROJECT_ROOT}/outputs/checkpoint_dynamics/scored"

EVAL_ROOT="${PROJECT_ROOT}/outputs/checkpoint_dynamics/evaluation_all"

LOG_ROOT="${PROJECT_ROOT}/logs/checkpoint_dynamics"


# ============================================================
# Fixed independent calibration data
# ============================================================

CALIB_RESPONSES="${CAUTION_ROOT}/outputs/repro_v2/normalization_gsm8k_train_heldout/all_responses.jsonl"

RND_MODEL="${CAUTION_ROOT}/outputs/repro_v2/repro_v2_rnd_gsm8k_s42_20260825_025440"

INFERENCE_CONFIG="${PROJECT_ROOT}/configs/llama32_3b_generation.json"


# ============================================================
# Create output directories
# ============================================================

mkdir -p \
    "$NORMALIZATION_ROOT" \
    "$SCORE_ROOT" \
    "$EVAL_ROOT" \
    "$LOG_ROOT"


cd "$PROJECT_ROOT"

export PYTHONPATH=.


# ============================================================
# Logging
# ============================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_FILE="${LOG_ROOT}/checkpoint_eval_all_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1


echo
echo "============================================================"
echo "Checkpoint evaluation"
echo "============================================================"
echo "GPU              : ${GPU}"
echo "Epochs           : ${EPOCHS[*]}"
echo "Datasets         : ${DATASETS[*]}"
echo "Trials           : ${TRIALS[*]}"
echo "MC samples       : ${NUM_MC}"
echo "NeuBoots lambda  : ${LAMBDA}"
echo "Log              : ${LOG_FILE}"
echo "============================================================"
echo


# ============================================================
# Input path
#
# These are the prepared JSONL files:
#
#     instance_id
#     prompt
#     all_samples
# ============================================================

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


# ============================================================
# Locate Caution detailed_candidates.json
# ============================================================

get_caution_detail_path() {

    local dataset="$1"
    local trial="$2"

    local directory="${CAUTION_EVAL_ROOT}/${dataset}/trial${trial}"

    if [[ ! -d "$directory" ]]; then
        echo "Missing Caution directory: ${directory}" >&2
        exit 1
    fi

    mapfile -t matches < <(
        find "$directory" \
            -maxdepth 1 \
            -type f \
            -name "*detailed_candidates.json" \
            | sort
    )

    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "No detailed_candidates.json found in:" >&2
        echo "  ${directory}" >&2
        exit 1
    fi

    if [[ "${#matches[@]}" -gt 1 ]]; then
        echo "Multiple detailed candidate files found:" >&2

        printf '  %s\n' "${matches[@]}" >&2

        exit 1
    fi

    echo "${matches[0]}"
}


# ============================================================
# Scored output path
#
# Reuse the GSM8K trial1 scoring we already started.
# score_gsm8k_neuboots.py resumes completed instance IDs.
# ============================================================

get_score_path() {

    local dataset="$1"
    local trial="$2"
    local epoch="$3"

    local e
    e=$(printf "%03d" "$epoch")

    if [[ "$dataset" == "gsm8k" && "$trial" == "1" ]]; then

        echo "${PROJECT_ROOT}/outputs/checkpoint_dynamics/gsm8k_trial1_scores/epoch_${e}.jsonl"

    else

        echo "${SCORE_ROOT}/${dataset}/trial${trial}/epoch_${e}.jsonl"

    fi
}


# ============================================================
# Preflight
# ============================================================

echo
echo "============================================================"
echo "PRE-FLIGHT CHECK"
echo "============================================================"


if [[ ! -f "$CALIB_RESPONSES" ]]; then
    echo "Missing calibration responses:"
    echo "$CALIB_RESPONSES"
    exit 1
fi


if [[ ! -d "$RND_MODEL" ]]; then
    echo "Missing RND model:"
    echo "$RND_MODEL"
    exit 1
fi


for epoch in "${EPOCHS[@]}"
do

    E=$(printf "%03d" "$epoch")

    CHECKPOINT_DIR="${CHECKPOINT_ROOT}/epoch_${E}"

    if [[ ! -f "${CHECKPOINT_DIR}/reward_predictor.pt" ]]; then
        echo "Missing checkpoint:"
        echo "${CHECKPOINT_DIR}/reward_predictor.pt"
        exit 1
    fi

    if [[ ! -f "${CHECKPOINT_DIR}/reward_predictor_config.json" ]]; then
        echo "Missing config:"
        echo "${CHECKPOINT_DIR}/reward_predictor_config.json"
        exit 1
    fi

done


for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        INPUT=$(get_input_path "$dataset" "$trial")

        if [[ ! -f "$INPUT" ]]; then

            echo
            echo "Missing prepared input:"
            echo "  dataset=${dataset}"
            echo "  trial=${trial}"
            echo "  path=${INPUT}"
            echo

            exit 1
        fi

        DETAIL=$(get_caution_detail_path "$dataset" "$trial")

        echo "[OK] ${dataset} trial${trial}"
        echo "     input  : ${INPUT}"
        echo "     caution: ${DETAIL}"

    done

done


echo
echo "Pre-flight check passed."
echo


# ============================================================
# STEP 1
# Epoch-specific normalization
#
# One calibration per checkpoint.
#
# The same normalization stats are reused for:
#
#   GSM8K trial1/2/3
#   MATH500 trial1/2/3
#   BBH trial1/2/3
# ============================================================

echo
echo "============================================================"
echo "STEP 1 / 3 : CHECKPOINT NORMALIZATION"
echo "============================================================"


for epoch in "${EPOCHS[@]}"
do

    E=$(printf "%03d" "$epoch")

    CHECKPOINT_DIR="${CHECKPOINT_ROOT}/epoch_${E}"

    CALIB_SCORES="${NORMALIZATION_ROOT}/epoch_${E}_calibration_scores.jsonl"

    NORMALIZATION_STATS="${NORMALIZATION_ROOT}/epoch_${E}_normalization_stats.json"


    echo
    echo "------------------------------------------------------------"
    echo "Epoch ${epoch}"
    echo "------------------------------------------------------------"


    # --------------------------------------------------------
    # If stats already exist, calibration is complete.
    # --------------------------------------------------------

    if [[ -s "$NORMALIZATION_STATS" ]]; then

        echo "Normalization stats already exist."
        echo "Skipping epoch ${epoch} calibration."

        continue

    fi


    # --------------------------------------------------------
    # Remove potentially incomplete calibration file.
    # Calibration script itself is not resumable.
    # --------------------------------------------------------

    rm -f "$CALIB_SCORES"


    CUDA_VISIBLE_DEVICES="$GPU" \
    python -u \
        ensemble/evaluation/score_normalization_calibration.py \
        --responses-file "$CALIB_RESPONSES" \
        --inference-config "$INFERENCE_CONFIG" \
        --generation-output-dir \
            "${NORMALIZATION_ROOT}/unused_generation_epoch_${E}" \
        --rnd-model-path "$RND_MODEL" \
        --neuboots-checkpoint-dir "$CHECKPOINT_DIR" \
        --num-mc "$NUM_MC" \
        --batch-size "$BATCH_SIZE" \
        --device cuda \
        --seed "$SEED" \
        --output-path "$CALIB_SCORES"


    python \
        ensemble/evaluation/build_normalization_stats.py \
        --input-path "$CALIB_SCORES" \
        --output-path "$NORMALIZATION_STATS"


    echo
    echo "Created:"
    echo "  ${NORMALIZATION_STATS}"

done


# ============================================================
# STEP 2
# Score every dataset / trial / checkpoint
#
# 3 datasets × 3 trials × 6 checkpoints
# = 54 scoring combinations
#
# Existing JSONL files are resumed automatically by
# score_gsm8k_neuboots.py.
# ============================================================

echo
echo "============================================================"
echo "STEP 2 / 3 : SCORE ALL CHECKPOINTS"
echo "============================================================"


for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        INPUT=$(get_input_path "$dataset" "$trial")


        for epoch in "${EPOCHS[@]}"
        do

            E=$(printf "%03d" "$epoch")

            CHECKPOINT_DIR="${CHECKPOINT_ROOT}/epoch_${E}"

            OUTPUT=$(get_score_path \
                "$dataset" \
                "$trial" \
                "$epoch"
            )

            mkdir -p "$(dirname "$OUTPUT")"


            echo
            echo "============================================================"
            echo "Scoring"
            echo "Dataset : ${dataset}"
            echo "Trial   : ${trial}"
            echo "Epoch   : ${epoch}"
            echo "Output  : ${OUTPUT}"
            echo "============================================================"


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


            echo -n "Scored problems: "
            wc -l < "$OUTPUT"

        done

    done

done


# ============================================================
# STEP 3
# BoN evaluation
#
# Each combination evaluates:
#
#   RM
#   NeuBoots pessimism only
#   RM + NeuBoots
#
# N:
#   1 2 4 8 16 32 64 128 256 512
# ============================================================

echo
echo "============================================================"
echo "STEP 3 / 3 : BoN EVALUATION"
echo "============================================================"


for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        DETAIL=$(get_caution_detail_path \
            "$dataset" \
            "$trial"
        )


        for epoch in "${EPOCHS[@]}"
        do

            E=$(printf "%03d" "$epoch")

            SCORE_FILE=$(get_score_path \
                "$dataset" \
                "$trial" \
                "$epoch"
            )

            NORMALIZATION_STATS="${NORMALIZATION_ROOT}/epoch_${E}_normalization_stats.json"

            OUTPUT_DIR="${EVAL_ROOT}/${dataset}/trial${trial}/epoch_${E}"

            mkdir -p "$OUTPUT_DIR"


            echo
            echo "============================================================"
            echo "Evaluation"
            echo "Dataset : ${dataset}"
            echo "Trial   : ${trial}"
            echo "Epoch   : ${epoch}"
            echo "============================================================"


            python \
                ensemble/evaluation/evaluate_fig3_methods.py \
                --dataset "$dataset" \
                --caution-detailed "$DETAIL" \
                --normalization-stats "$NORMALIZATION_STATS" \
                --neuboots-results "$SCORE_FILE" \
                --methods \
                    rm \
                    neuboots_pessimism \
                    rm_neuboots \
                --neuboots-lambda "$LAMBDA" \
                --num-bootstrap "$NUM_BOOTSTRAP" \
                --bootstrap-seed "$SEED" \
                --output-dir "$OUTPUT_DIR"

        done

    done

done


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "ALL CHECKPOINT EVALUATIONS COMPLETED"
echo "============================================================"
echo
echo "Normalization:"
echo "  ${NORMALIZATION_ROOT}"
echo
echo "Scores:"
echo "  ${SCORE_ROOT}"
echo
echo "Evaluation:"
echo "  ${EVAL_ROOT}"
echo
echo "Log:"
echo "  ${LOG_FILE}"
echo
