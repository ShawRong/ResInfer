#!/usr/bin/env bash
# End-to-end release smoke for the public ResInfer path.
#
# This script is intentionally small and deterministic:
#   1. export a tiny residue NVFP4 checkpoint with synthetic calibration data;
#   2. launch the repo's vLLM serving entrypoint against that checkpoint;
#   3. send one OpenAI-compatible chat request and verify a non-empty response.
#
# It is a wiring check, not a quality benchmark.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  GPU=0 RESINFER_SMOKE_PORT=8000 export/run_export_to_serving_smoke.sh

Environment overrides:
  RESINFER_SMOKE_MODEL_PATH      HF repo ID or local model directory
                                 default: Qwen/Qwen3-0.6B
  RESINFER_SMOKE_HOST            vLLM bind host; default: 127.0.0.1
  RESINFER_SMOKE_PORT            vLLM server port; default: 8000
  RESINFER_SMOKE_SERVED_NAME     OpenAI API model name; default: resinfer_smoke
  OUTPUT_DIR                     smoke output directory
  GPU                            CUDA_VISIBLE_DEVICES value
  RUNTIME_PY                     runtime venv python
  EXPORT_PY                      export venv python
  MAX_WAIT_SEC                   server readiness timeout
  MAX_MODEL_LEN                  vLLM max_model_len for smoke
  MAX_NUM_BATCHED_TOKENS         vLLM max_num_batched_tokens for smoke

This is a wiring smoke, not a quality benchmark.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[release-smoke] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_PY="${RUNTIME_PY:-$REPO_ROOT/.venv/bin/python}"
EXPORT_PY="${EXPORT_PY:-$REPO_ROOT/env-export/.venv/bin/python}"
MODEL_PATH="${RESINFER_SMOKE_MODEL_PATH:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/resinfer_release_smoke_$(date +%Y%m%d_%H%M%S)}"
HOST="${RESINFER_SMOKE_HOST:-127.0.0.1}"
PORT="${RESINFER_SMOKE_PORT:-8000}"
SERVED_MODEL_NAME="${RESINFER_SMOKE_SERVED_NAME:-resinfer_smoke}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

if [[ ! -x "$RUNTIME_PY" ]]; then
  echo "[release-smoke] missing runtime python: $RUNTIME_PY" >&2
  exit 1
fi
if [[ ! -x "$EXPORT_PY" ]]; then
  echo "[release-smoke] missing export python: $EXPORT_PY" >&2
  exit 1
fi
if [[ "$MODEL_PATH" = /* && ! -e "$MODEL_PATH" ]]; then
  echo "[release-smoke] local RESINFER_SMOKE_MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi
# The smoke talks to a local vLLM server. Do not let user/system HTTP proxy
# variables capture localhost requests.
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

mkdir -p "$OUTPUT_DIR"
EXPORT_LOG="$OUTPUT_DIR/export.log"
SERVER_LOG="$OUTPUT_DIR/server.log"
REQUEST_LOG="$OUTPUT_DIR/request.json"
EXPORT_DIR="$OUTPUT_DIR/export/exported_residue"

echo "[release-smoke] repo=$REPO_ROOT"
echo "[release-smoke] output=$OUTPUT_DIR"
echo "[release-smoke] model=$MODEL_PATH"

echo "[release-smoke] step 1/3: export residue checkpoint"
"$EXPORT_PY" "$REPO_ROOT/export/export_residue_model.py" \
  --model_path "$MODEL_PATH" \
  --dataset_name synthetic \
  --awq_samples 2 \
  --search_samples 2 \
  --nvfp4_samples 2 \
  --awq_seq_len 64 \
  --search_seq_len 64 \
  --nvfp4_seq_len 64 \
  --search_calib_steps 1 \
  --search_score_steps 1 \
  --ratios 0.125 \
  --budget 4.5 \
  --output_dir "$OUTPUT_DIR/export" \
  >"$EXPORT_LOG" 2>&1

if [[ ! -f "$EXPORT_DIR/residue_kernel_metadata.json" ]]; then
  echo "[release-smoke] missing exported residue metadata under $EXPORT_DIR" >&2
  tail -80 "$EXPORT_LOG" >&2 || true
  exit 1
fi

echo "[release-smoke] step 2/3: start vLLM server"
# FlashInfer sampler JIT can be fragile across CUDA/SM combinations. This smoke
# validates the ResInfer linear path, so keep sampling on the stable vLLM path
# unless the caller explicitly opts in.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

"$RUNTIME_PY" "$REPO_ROOT/.dev/tools/vllm_patch/serve_vllm.py" \
  --model_path "$EXPORT_DIR" \
  --served_model_name "$SERVED_MODEL_NAME" \
  --quantization residue_nvfp4 \
  --residue_patch_mode mext_r1 \
  --enable_decode_mext \
  --max_model_len "$MAX_MODEL_LEN" \
  --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
  --host "$HOST" \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$RUNTIME_PY" - "$HOST" "$PORT" "$MAX_WAIT_SEC" <<'PY'
import sys
import time
import urllib.request

host, port, max_wait = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
url = f"http://{host}:{port}/v1/models"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.time() + max_wait
last_error = None
while time.time() < deadline:
    try:
        with opener.open(url, timeout=5) as response:
            if response.status == 200:
                print(f"[release-smoke] server ready: {url}", flush=True)
                raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(2)
raise TimeoutError(f"server not ready after {max_wait}s: {last_error!r}")
PY

echo "[release-smoke] step 3/3: send chat completion"
"$RUNTIME_PY" - "$HOST" "$PORT" "$SERVED_MODEL_NAME" "$REQUEST_LOG" <<'PY'
import json
import sys
import urllib.request

host, port, model, request_log = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
url = f"http://{host}:{port}/v1/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Solve 17 + 25. Return only the number."}],
    "max_tokens": 32,
    "temperature": 0.0,
}
data = json.dumps(payload).encode("utf-8")
request = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=120) as response:
    body = response.read().decode("utf-8")
result = json.loads(body)
content = result["choices"][0]["message"]["content"]
if not str(content).strip():
    raise AssertionError(f"empty completion: {result!r}")
with open(request_log, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)
print("[release-smoke] completion:", content.strip(), flush=True)
PY

cat >"$OUTPUT_DIR/release_smoke_summary.json" <<JSON
{
  "status": "PASS",
  "model_path": "$MODEL_PATH",
  "export_dir": "$EXPORT_DIR",
  "served_model_name": "$SERVED_MODEL_NAME",
  "server_log": "$SERVER_LOG",
  "export_log": "$EXPORT_LOG",
  "request_log": "$REQUEST_LOG"
}
JSON

echo "[release-smoke] PASS"
echo "[release-smoke] summary=$OUTPUT_DIR/release_smoke_summary.json"
