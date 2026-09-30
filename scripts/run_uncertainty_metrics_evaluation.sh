#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Settings
# ============================================================

LAMBDA=0.8
NUM_BOOTSTRAP=1000
SEED=42

DATASETS=(gsm8k math500 bbh)
TRIALS=(1 2 3)

QUANTILE_TYPES=(
    median_quantile
    avg_quantile
    avg_quantile_avg
)

ALPHAS=(
    0.0
    0.1
    0.2
    0.3
    0.4
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT="$HOME/workplace/caution_based_ensemble"

CAUTION_ROOT="$HOME/workplace/Caution"

CAUTION_EVAL_ROOT="${CAUTION_ROOT}/outputs/repro_v2/repro_v2_eval_20260825_173851"

NORMALIZATION_STATS="${PROJECT_ROOT}/outputs/uncertainty_metrics/normalization/normalization_stats.json"

SCORE_ROOT="${PROJECT_ROOT}/outputs/uncertainty_metrics/scored_5metrics"

EVAL_ROOT="${PROJECT_ROOT}/outputs/uncertainty_metrics/evaluation_5metrics"

LOG_ROOT="${PROJECT_ROOT}/logs/uncertainty_metrics"


mkdir -p \
    "$EVAL_ROOT" \
    "$LOG_ROOT"


cd "$PROJECT_ROOT"

export PYTHONPATH=.


# ============================================================
# Logging
# ============================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_FILE="${LOG_ROOT}/evaluation_5metrics_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1


echo
echo "============================================================"
echo "NeuBoots uncertainty metric evaluation"
echo "============================================================"
echo "Lambda            : ${LAMBDA}"
echo "Normalization     : ${NORMALIZATION_STATS}"
echo "Score root        : ${SCORE_ROOT}"
echo "Evaluation root   : ${EVAL_ROOT}"
echo "============================================================"
echo


# ============================================================
# Find Caution detailed candidate file
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
        echo "No detailed_candidates file found in:" >&2
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
# Run one evaluation
# ============================================================

run_eval() {

    local dataset="$1"
    local trial="$2"
    local metric="$3"
    local alpha="${4:-}"

    local DETAIL
    DETAIL=$(get_caution_detail_path \
        "$dataset" \
        "$trial"
    )

    local SCORE_FILE="${SCORE_ROOT}/${dataset}/trial${trial}.jsonl"

    if [[ ! -f "$SCORE_FILE" ]]; then
        echo "Missing score file:"
        echo "  ${SCORE_FILE}"
        exit 1
    fi

    local CONFIG_NAME

    if [[ -n "$alpha" ]]; then
        CONFIG_NAME="${metric}_alpha_${alpha}"
    else
        CONFIG_NAME="${metric}"
    fi

    local OUTPUT_DIR="${EVAL_ROOT}/${dataset}/trial${trial}/${CONFIG_NAME}"

    mkdir -p "$OUTPUT_DIR"

    echo
    echo "============================================================"
    echo "Evaluation"
    echo "Dataset : ${dataset}"
    echo "Trial   : ${trial}"
    echo "Metric  : ${metric}"

    if [[ -n "$alpha" ]]; then
        echo "Alpha   : ${alpha}"
    fi

    echo "Output  : ${OUTPUT_DIR}"
    echo "============================================================"

    CMD=(
        python -u
        ensemble/evaluation/evaluate_fig3_methods.py
        --dataset "$dataset"
        --caution-detailed "$DETAIL"
        --normalization-stats "$NORMALIZATION_STATS"
        --neuboots-results "$SCORE_FILE"
        --methods rm_neuboots
        --neuboots-lambda "$LAMBDA"
        --neuboots-uncertainty-type "$metric"
        --num-bootstrap "$NUM_BOOTSTRAP"
        --bootstrap-seed "$SEED"
        --output-dir "$OUTPUT_DIR"
    )

    if [[ -n "$alpha" ]]; then
        CMD+=(
            --neuboots-alpha "$alpha"
        )
    fi

    "${CMD[@]}"
}


# ============================================================
# Preflight
# ============================================================

if [[ ! -f "$NORMALIZATION_STATS" ]]; then
    echo "Missing normalization stats:"
    echo "  ${NORMALIZATION_STATS}"
    exit 1
fi


for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        SCORE_FILE="${SCORE_ROOT}/${dataset}/trial${trial}.jsonl"

        if [[ ! -f "$SCORE_FILE" ]]; then
            echo "Missing NeuBoots score file:"
            echo "  ${SCORE_FILE}"
            exit 1
        fi

        DETAIL=$(get_caution_detail_path \
            "$dataset" \
            "$trial"
        )

        echo "[OK] ${dataset} trial${trial}"
        echo "     score   : ${SCORE_FILE}"
        echo "     caution : ${DETAIL}"

    done

done


echo
echo "Pre-flight check passed."
echo


# ============================================================
# Evaluation
# ============================================================

for dataset in "${DATASETS[@]}"
do

    for trial in "${TRIALS[@]}"
    do

        # ----------------------------------------------------
        # 1. STD
        # ----------------------------------------------------

        run_eval \
            "$dataset" \
            "$trial" \
            "std"


        # ----------------------------------------------------
        # 2. Distance
        # ----------------------------------------------------

        run_eval \
            "$dataset" \
            "$trial" \
            "distance"


        # ----------------------------------------------------
        # 3-5. Quantile metrics
        # ----------------------------------------------------

        for metric in "${QUANTILE_TYPES[@]}"
        do

            for alpha in "${ALPHAS[@]}"
            do

                run_eval \
                    "$dataset" \
                    "$trial" \
                    "$metric" \
                    "$alpha"

            done

        done

    done

done


echo
echo "============================================================"
echo "ALL UNCERTAINTY EVALUATIONS COMPLETED"
echo "============================================================"
echo
echo "Output:"
echo "  ${EVAL_ROOT}"
echo
echo "Log:"
echo "  ${LOG_FILE}"
echo
