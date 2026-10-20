#!/usr/bin/env bash
# Run from a verified Pod-local checkout with an already validated export environment.
# Optional first argument selects one model; otherwise export all four in order.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${EXPORT_PYTHON:?Set EXPORT_PYTHON to the validated export interpreter}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT containing the four source model directories}"
DATASET="${CALIBRATION_DATASET:?Set CALIBRATION_DATASET to calibration.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new Pod-local campaign directory}"
SELECTED="${1:-all}"
case "$SELECTED" in all|qwen3-4b|qwen3-8b|qwen3-32b|qwen35-27b) ;; *) echo "Unknown model: $SELECTED" >&2; exit 2;; esac


export_model() {
    local model="$1"
    [[ "$SELECTED" == all || "$SELECTED" == "$model" ]] || return 0
    mkdir -p "$OUTPUT_ROOT/$model/nvfp4" "$OUTPUT_ROOT/$model/mxfp4"
    # One recipe across formats/ratios; each checkpoint uses its own tokenizer.
    local common=(--source "$MODEL_ROOT/$model" --dataset "$DATASET"
        --samples 1024 --seq-len 2048 --batch-size 8 --seed 42
        --reasoning-effort medium)
    "$PYTHON" -m export.export_nvfp4 "${common[@]}" \
        --mode preflight --output "$OUTPUT_ROOT/$model/preflight"
    for format in nvfp4 mxfp4; do
        "$PYTHON" -m export.export_nvfp4 "${common[@]}" --format "$format" \
            --mode plain --output "$OUTPUT_ROOT/$model/$format/plain"
        "$PYTHON" -m export.export_nvfp4 "${common[@]}" --format "$format" \
            --mode residue --ratio 0.125 --output "$OUTPUT_ROOT/$model/$format/r0125"
        "$PYTHON" -m export.export_nvfp4 "${common[@]}" --format "$format" \
            --mode residue --ratio 0.25 --output "$OUTPUT_ROOT/$model/$format/r025"
        "$PYTHON" -m export.export_nvfp4 "${common[@]}" --format "$format" \
            --mode residue --ratio 0.5 --output "$OUTPUT_ROOT/$model/$format/r05"
        "$PYTHON" -m export.export_nvfp4 "${common[@]}" --format "$format" \
            --mode residue --ratio 1.0 --output "$OUTPUT_ROOT/$model/$format/r1"
    done
}

export_model qwen3-4b
export_model qwen3-8b
export_model qwen3-32b
export_model qwen35-27b
# complete.json currently certifies structure only; run serving acceptance before publishing.
