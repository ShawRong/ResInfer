'Build (do not launch) a real-model, attention-only vLLM A/B experiment.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def build_plan(model: str, config: dict, *, python: str, gpu: str,
               input_len: int, output_len: int, concurrency: int,
               max_model_len: int, tp: int = 1, eager: bool = False,
               offline_cubin_dir: str | None = None,
               kv_cache_bytes: int | None = None,
               attn_quant_fusion: bool = False) -> dict:
    """Single common model/settings; only attention selection varies by arm."""
    if min(input_len, output_len, concurrency, max_model_len, tp) <= 0:
        raise ValueError("lengths, concurrency and TP must be positive")
    cfg = config.get("text_config", config)
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    dim = cfg.get("head_dim", cfg["hidden_size"] // hq)
    if hkv <= 0 or hq % hkv or dim != 128 or hq // hkv not in (4, 6, 8):
        raise ValueError("QK-residue dispatch requires D128 and GQA4/6/8")
    if hq % tp or hkv % tp:
        raise ValueError("this initial experiment requires TP to divide Q/KV heads")
    devices = gpu.split(",")
    if len(devices) != tp or len(set(devices)) != tp or not all(
        d.isdecimal() for d in devices
    ):
        raise ValueError("provide exactly TP distinct physical GPU indices")
    if input_len + output_len > max_model_len:
        raise ValueError("input + generated tokens exceed max_model_len")
    if max_model_len > cfg["max_position_embeddings"]:
        raise ValueError("do not silently extend the checkpoint context limit")
    if kv_cache_bytes is not None and kv_cache_bytes <= 0:
        raise ValueError("kv_cache_bytes must be positive")
    if eager and attn_quant_fusion:
        raise ValueError("attention output fusion requires a compiled experiment")
    common = [python, "-m", "vllm.entrypoints.cli.main", "serve", model,
              "--served-model-name", "resattn-e2e", "--host", "127.0.0.1",
              "--port", "8123", "--dtype", "bfloat16",
              "--tensor-parallel-size", str(tp), "--kv-cache-dtype", "nvfp4",
              "--block-size", "128", "--max-model-len", str(max_model_len),
              "--max-num-seqs", str(concurrency), "--max-num-batched-tokens",
              str(max(2048, concurrency)), "--gpu-memory-utilization", "0.85",
              "--no-enable-prefix-caching", "--enable-chunked-prefill",
              "--seed", "17"]
    if eager:
        common.append("--enforce-eager")
    if kv_cache_bytes is not None:
        common += ["--kv-cache-memory-bytes", str(kv_cache_bytes)]
    if attn_quant_fusion:
        common += ["--compilation-config", json.dumps(
            {"pass_config": {"fuse_attn_quant": True}})]
    common_env = {"CUDA_VISIBLE_DEVICES": gpu,
                  "VLLM_DISABLE_COMPILE_CACHE": "1"}
    if offline_cubin_dir:
        # vLLM's availability probe otherwise contacts NVIDIA even when a
        # FLASHINFER_CUBIN_DIR exists. Actual artifact SHA checks stay enabled.
        common_env.update(VLLM_HAS_FLASHINFER_CUBIN="1",
                          FLASHINFER_NO_DOWNLOAD="1",
                          FLASHINFER_CUBIN_DIR=offline_cubin_dir)
    # Start both arms from a clean environment so inherited linear and
    # attention settings cannot change this attention-only comparison.
    unset = sorted(k for k in os.environ if k.startswith(("RESIDUE_", "NVFP4_")))
    unset = sorted(set(unset + ["VLLM_PLUGINS", "VLLM_ATTENTION_BACKEND",
                               "VLLM_BATCH_INVARIANT"]))
    arms = {}
    for arm in ("stock", "resattn"):
        env = dict(common_env)
        if arm == "resattn":
            env.update(RESIDUE_ATTN="1", RESIDUE_ATTN_REDUCTION="auto")
            if not eager:
                env["RESIDUE_ATTN_CUDAGRAPH"] = "1"
        arms[arm] = {"env_set": env, "env_unset": unset,
                     "server_argv": (
                         common + ["--attention-backend", "CUSTOM"]
                         if arm == "resattn" else list(common)
                     )}
    bench = [python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
             "--backend", "vllm", "--base-url", "http://127.0.0.1:8123",
             "--model", "resattn-e2e", "--tokenizer", model,
             "--dataset-name", "random", "--random-input-len", str(input_len),
             "--random-output-len", str(output_len), "--random-range-ratio", "0.0",
             "--num-prompts", str(max(32, concurrency * 8)),
             "--num-warmups", "8",
             "--max-concurrency", str(concurrency), "--request-rate", "inf",
             "--ignore-eos", "--temperature", "0", "--seed", "17",
             "--save-result", "--save-detailed",
             "--percentile-metrics", "ttft,tpot,itl,e2el"]
    return {
        "status": "PLAN_ONLY_NOT_RUN", "model": model,
        "model_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "geometry": {"head_dim": dim, "q_heads": hq, "kv_heads": hkv},
        "output_contract": "nvfp4" if attn_quant_fusion else "bf16",
        "mode": "eager_smoke" if eager else "graph_candidate_after_smoke",
        "arms": arms, "benchmark_argv": bench,
        "round_orders": [["stock", "resattn"], ["resattn", "stock"]] * 2,
        "launch_requirements": [
            "Explicit model/task selection and freshly verified free GPU ownership",
            "Isolated validated Runtime, project plugin and hashed source snapshot",
            "Confirm the residue-vllm-plugin entry point exists in spawned workers",
            "Unset all RESIDUE_*/NVFP4_* vars at execution time before env_set",
            "Inspect other inherited vLLM overrides; keep identical across arms",
            "Verify installed CLI flags before launch; this file does not execute",
            "Record the actual vLLM-selected stock backend and kernel, not an assumed TRT label",
            "Diagnostic proof of stock decode and CUSTOM QK residue engagement",
            "Both arms use stock FP4 context; verify output fusion and graph replay",
            "For NVFP4 output, verify the downstream O-projection consumes packed data and scales",
            "Fix equal KV capacity before formal A/B; log capacity/preemptions",
            "Warm up outside timed requests; record actual batch/length distribution",
            "Use fresh per-round result directories; run arms sequentially",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--eager", action="store_true", help="initial smoke only")
    ap.add_argument("--offline-cubin-dir", help="prepopulated, verified TRT artifact cache")
    ap.add_argument("--kv-cache-bytes", type=int, help="same explicit KV capacity per arm")
    ap.add_argument("--attn-quant-fusion", action="store_true")
    args = ap.parse_args()
    config = json.loads((Path(args.model) / "config.json").read_text())
    print(json.dumps(build_plan(args.model, config, python=sys.executable,
                               gpu=args.gpu, input_len=args.input_len,
                               output_len=args.output_len,
                               concurrency=args.concurrency,
                               max_model_len=args.max_model_len, tp=args.tp,
                               eager=args.eager,
                               offline_cubin_dir=args.offline_cubin_dir,
                               kv_cache_bytes=args.kv_cache_bytes,
                               attn_quant_fusion=args.attn_quant_fusion), indent=2))


if __name__ == "__main__":
    main()
