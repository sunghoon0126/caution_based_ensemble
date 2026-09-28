#!/usr/bin/env bash

set -u

ROOT="$HOME/workplace/caution_based_ensemble"
OUT="$ROOT/outputs"

echo
echo "======================================================================"
echo "1. ALL TRAINED REWARD PREDICTOR CHECKPOINTS"
echo "======================================================================"

find "$OUT" \
    -type f \
    -name "reward_predictor.pt" \
    2>/dev/null \
    | sort

echo
echo
echo "======================================================================"
echo "2. TRAINING HISTORY FILES"
echo "======================================================================"

find "$OUT" \
    -type f \
    -name "training_history.csv" \
    2>/dev/null \
    | sort

echo
echo
echo "======================================================================"
echo "3. CHECKPOINT DIRECTORIES"
echo "======================================================================"

find "$OUT" \
    -type d \
    -name "epoch_*" \
    2>/dev/null \
    | sort

echo
echo
echo "======================================================================"
echo "4. KNOWN GSM8K CHECKPOINT DYNAMICS"
echo "======================================================================"

GSM_CKPT="$OUT/checkpoint_dynamics/gsm8k_lightweight/checkpoints"

for E in 000 001 002 003 004 005
do
    FILE="$GSM_CKPT/epoch_${E}/reward_predictor.pt"

    if [[ -f "$FILE" ]]; then
        echo "epoch_${E}: OK"
    else
        echo "epoch_${E}: MISSING"
    fi
done

echo
echo
echo "======================================================================"
echo "5. PREPARED EVALUATION INPUTS"
echo "======================================================================"

echo
echo "[GSM8K]"

for T in 1 2 3
do
    FILE="$OUT/table3_repro/baseline/trial${T}_input.jsonl"

    if [[ -f "$FILE" ]]; then
        echo -n "trial${T}: "
        wc -l < "$FILE"
    else
        echo "trial${T}: MISSING"
    fi
done

echo
echo "[MATH500]"

for T in 1 2 3
do
    FILE="$OUT/fig3_neuboots/math500/trial${T}_input.jsonl"

    if [[ -f "$FILE" ]]; then
        echo -n "trial${T}: "
        wc -l < "$FILE"
    else
        echo "trial${T}: MISSING"
    fi
done

echo
echo "[BBH]"

for T in 1 2 3
do
    FILE="$OUT/fig3_neuboots/bbh/trial${T}_input.jsonl"

    if [[ -f "$FILE" ]]; then
        echo -n "trial${T}: "
        wc -l < "$FILE"
    else
        echo "trial${T}: MISSING"
    fi
done

echo
echo
echo "======================================================================"
echo "6. GSM8K CHECKPOINT SCORING STATUS"
echo "======================================================================"

for E in 000 001 002 003 004 005
do

    FILE="$OUT/checkpoint_dynamics/gsm8k_trial1_scores/epoch_${E}.jsonl"

    if [[ -f "$FILE" ]]; then
        echo -n "trial1 epoch_${E}: "
        wc -l < "$FILE"
    else
        echo "trial1 epoch_${E}: MISSING"
    fi

done

for T in 2 3
do

    for E in 000 001 002 003 004 005
    do

        FILE="$OUT/checkpoint_dynamics/scored/gsm8k/trial${T}/epoch_${E}.jsonl"

        if [[ -f "$FILE" ]]; then
            echo -n "trial${T} epoch_${E}: "
            wc -l < "$FILE"
        else
            echo "trial${T} epoch_${E}: MISSING"
        fi

    done

done

echo
echo
echo "======================================================================"
echo "7. MATH500 CHECKPOINT SCORING STATUS"
echo "======================================================================"

for T in 1 2 3
do

    for E in 000 001 002 003 004 005
    do

        FILE="$OUT/checkpoint_dynamics/scored/math500/trial${T}/epoch_${E}.jsonl"

        if [[ -f "$FILE" ]]; then
            echo -n "trial${T} epoch_${E}: "
            wc -l < "$FILE"
        else
            echo "trial${T} epoch_${E}: MISSING"
        fi

    done

done

echo
echo
echo "======================================================================"
echo "8. BBH CHECKPOINT SCORING STATUS"
echo "======================================================================"

for T in 1 2 3
do

    for E in 000 001 002 003 004 005
    do

        FILE="$OUT/checkpoint_dynamics/scored/bbh/trial${T}/epoch_${E}.jsonl"

        if [[ -f "$FILE" ]]; then
            echo -n "trial${T} epoch_${E}: "
            wc -l < "$FILE"
        else
            echo "trial${T} epoch_${E}: MISSING"
        fi

    done

done

echo
echo
echo "======================================================================"
echo "9. NORMALIZATION STATS"
echo "======================================================================"

find "$OUT" \
    -type f \
    \( \
        -name "*normalization_stats*.json" \
        -o \
        -name "normalization_stats.json" \
    \) \
    2>/dev/null \
    | sort

echo
echo
echo "======================================================================"
echo "10. EVALUATION OUTPUTS"
echo "======================================================================"

find "$OUT/checkpoint_dynamics" \
    -type f \
    \( \
        -name "*.csv" \
        -o \
        -name "*.json" \
    \) \
    2>/dev/null \
    | grep -E "evaluation|epoch_|summary" \
    | sort

echo
echo
echo "======================================================================"
echo "11. DATASET-NAMED TRAINING OUTPUT DIRECTORIES"
echo "======================================================================"

find "$OUT" \
    -maxdepth 5 \
    -type d \
    2>/dev/null \
    | grep -Ei "gsm8k|math500|math_500|math-500|bbh" \
    | sort

echo
echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
