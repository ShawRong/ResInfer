#!/usr/bin/env bash
# Serial quality campaign; run from Pod-local storage with a validated runtime.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON="${EVAL_PYTHON:?Set EVAL_PYTHON to the inference interpreter}"
MODEL_ROOT="${MODEL_ROOT:?BF16 directories named by model}"
EXPORT_ROOT="${EXPORT_ROOT:?Export directory containing model/format/arm/checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Use a new local experiment directory}"
TASK="${1:?Usage: bash benchmarks/quality/run_quality_models.sh ppl|lcb|aime [model|all]}"
SELECTED="${2:-all}"
case "$SELECTED" in all|qwen3-4b|qwen3-8b|qwen3-32b|qwen35-27b) ;; *) echo "Unknown model: $SELECTED" >&2; exit 2;; esac
case "$TASK" in
    ppl) : "${WIKITEXT_TASK_PATH:?Frozen lm-eval task YAML directory}" ;;
    lcb) : "${PROTOCOL_ROOT:?Directory of model-specific YAML manifests}"
         : "${LCB_SOURCE:?Pinned official LCB source}" "${LCB_DATA:?Frozen complete cohort}" ;;
    aime) : "${PROTOCOL_ROOT:?Directory of model-specific YAML manifests}"
          : "${LIGHTEVAL_ENV:?Pinned lighteval client environment}" ;;
    *) echo "Unknown task: $TASK (ppl, lcb, aime)" >&2; exit 2 ;;
esac

# Arm gates are resolved inside each fresh process, not by sitecustomize at startup.
export RESIDUE_NVFP4=0 RESIDUE_MXFP4=0 RESIDUE_ATTN=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT/$TASK"

run_command() {
    printf '%q ' "$@" >> "$case_out/commands.sh"
    printf '\n' >> "$case_out/commands.sh"
    "$@" 2>&1 | tee "$case_out/$arm.log"
}

run_arm() {
    local arm="$1" checkpoint="$2" kind="$3"
    local common=(--max-num-seqs "${MAX_NUM_SEQS:-8}" --gpu-mem-util "${GPU_MEM_UTIL:-0.85}")
    case "$TASK" in
        ppl)
            local command=("$PYTHON" -B -m benchmarks.quality.evalkit task
                --model "$checkpoint" --arm "$arm" --out "$case_out/results"
                --tasks wikitext --task-path "$WIKITEXT_TASK_PATH"
                --max-model-len "${PPL_MAX_MODEL_LEN:-4096}" "${common[@]}")
            case "$kind" in
                bf16) command+=(--bf16) ;;
                modelopt_fp4) command+=(--plain) ;;
                modelopt_mxfp4) command+=(--quantization modelopt_mxfp4) ;;
            esac
            [[ -z "${ENGINE_CONFIG:-}" ]] || command+=(--engine-config "$case_out/engine.json")
            run_command "${command[@]}"
            ;;
        lcb)
            local command=("$PYTHON" -B -m benchmarks.quality.lcb run
                --manifest "$case_out/manifest.yaml" --models-root "$MODEL_ROOT"
                --arms "$arm" --out "$case_out/results" --lcb-source "$LCB_SOURCE"
                --data "$LCB_DATA" --scorer-python "${SCORER_PYTHON:-$PYTHON}"
                --no-flashinfer-autotune --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
                "${common[@]}")
            [[ -z "${ENGINE_CONFIG:-}" ]] || command+=(--engine-config "$case_out/engine.json")
            run_command "${command[@]}"
            ;;
        aime)
            local command=("$PYTHON" -B -m benchmarks.quality.lighteval
                --manifest "$case_out/manifest.yaml" --models-root "$MODEL_ROOT"
                --arms "$arm" --out "$case_out/results/$arm" --lighteval-env "$LIGHTEVAL_ENV"
                --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
                --concurrent-requests "${MAX_NUM_SEQS:-8}" "${common[@]}")
            [[ -z "${CUSTOM_TASKS:-}" ]] || command+=(--custom-tasks "$CUSTOM_TASKS")
            [[ -z "${DATA_ROOT:-}" ]] || command+=(--data-root "$DATA_ROOT")
            [[ -z "${NLTK_DATA:-}" ]] || command+=(--nltk-data "$NLTK_DATA")
            run_command "${command[@]}"
            ;;
    esac
}

run_model() {
    local model="$1"
    [[ "$SELECTED" == all || "$SELECTED" == "$model" ]] || return 0
    local case_out="$OUTPUT_ROOT/$TASK/$model" arm checkpoint kind
    mkdir "$case_out"
    cp benchmarks/quality/run_quality_models.sh "$case_out/entry.sh"
    if [[ -n "${ENGINE_CONFIG:-}" ]]; then
        [[ "$TASK" != aime ]] || { echo "lighteval does not accept ENGINE_CONFIG" >&2; exit 2; }
        cp "$ENGINE_CONFIG" "$case_out/engine.json"
    fi
    "$PYTHON" - "$TASK" "${PROTOCOL_ROOT:-}/$model.yaml" "$MODEL_ROOT/$model" \
        "$EXPORT_ROOT/$model" "$case_out/manifest.yaml" > "$case_out/arms.tsv" <<'PYTHON'
import sys
from pathlib import Path

task = sys.argv[1]
protocol, bf16, exports, output = map(Path, sys.argv[2:])
arms = [{"id": "bf16", "path": str(bf16.resolve(strict=True)), "kind": "bf16"}]
for fmt in ("nvfp4", "mxfp4"):
    for arm in ("plain", "r0125", "r025", "r05", "r1"):
        path = (exports / fmt / arm / "checkpoint").resolve(strict=True)
        kind = "modelopt_mxfp4" if fmt == "mxfp4" else "modelopt_fp4" if arm == "plain" else "residue"
        arms.append({"id": f"{fmt}_{arm}", "path": str(path), "kind": kind})
if task != "ppl":
    import yaml

    manifest = yaml.safe_load(protocol.read_text())
    manifest["arms"] = arms
    with output.open("x") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
for arm in arms:
    print(arm["id"], arm["path"], arm["kind"], sep="\t")
PYTHON
    while IFS=$'\t' read -r arm checkpoint kind; do
        run_arm "$arm" "$checkpoint" "$kind" < /dev/null
    done < "$case_out/arms.tsv"
}

# 4B is opt-in; select it explicitly.
if [[ "$SELECTED" == qwen3-4b ]]; then run_model qwen3-4b; fi
run_model qwen3-8b
run_model qwen3-32b
run_model qwen35-27b
