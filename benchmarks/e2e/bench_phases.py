"""Full-model prefill or decode timing with real KV and a preparation barrier.

Run once per arm/process. TP=1, synchronous vLLM V1; decode starts only after
all prompts finish prefill. This is an isolated phase experiment, not serving.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from benchmarks.checkpoints import detect_quantization, validate_context
from benchmarks._runtime import assert_gpu_exclusive, gpu_state
from benchmarks.quality.common.runtime import write_effective_engine_config


def configure(args):
    quant = detect_quantization(args.model)
    if args.linear == "stock" and quant == "residue_nvfp4":
        raise ValueError("Stock requires a plain checkpoint, not extended-K weights")
    if args.linear == "residue" and quant != "residue_nvfp4":
        raise ValueError("Residue requires a checkpoint with residue metadata")
    if args.linear == "mext" and quant != "modelopt_fp4":
        raise ValueError("Pure MEXT requires a plain NVFP4 checkpoint")
    if quant == "modelopt_mxfp4":
        raise ValueError("This phase protocol currently covers BF16 and NVFP4 weights")
    if args.attention == "resattn" and args.kv != "nvfp4":
        raise ValueError("ResAttn requires --kv nvfp4")
    if args.phase == "prefill" and args.attention != "stock":
        raise ValueError("ResAttn is a decode comparison; use stock for prefill")
    validate_context(args.model, args.input_len + args.output_len)
    for key in list(os.environ):
        if key.startswith(("RESIDUE_", "RESINFER_DEV_")):
            del os.environ[key]
    os.environ.update(
        RESIDUE_NVFP4="1" if args.linear != "stock" else "0",
        RESIDUE_MXFP4="0", RESIDUE_ATTN="1" if args.attention == "resattn" else "0",
        RESIDUE_PATCH_MODE="mext_r1" if args.linear == "mext" else "v2",
        RESIDUE_ENABLE_DECODE_MEXT="1" if args.linear != "stock" else "0",
        RESIDUE_ATTN_CUDAGRAPH="1", RESIDUE_ATTN_REDUCTION="auto",
        VLLM_PLUGINS="", VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_NO_USAGE_STATS="1",
    )
    if args.linear != "stock":
        os.environ["RESIDUE_VLLM_MODEL_DIR"] = str(args.model)
    return quant


@contextmanager
def phase_trace(engine, graph_class):
    """Park prefilled requests until the full submitted batch can decode."""
    scheduler = engine.engine_core.engine_core.scheduler
    original_schedule, original_step = scheduler.schedule, engine.step
    original_replay = graph_class.replay
    parked, phases, steps = [], [], []
    preparing, replays = True, 0

    def replay(graph):
        nonlocal replays
        replays += 1
        return original_replay(graph)

    def schedule(*args, **kwargs):
        nonlocal preparing
        if preparing:
            pending = [r for r in scheduler.running if r.num_computed_tokens < r.num_prompt_tokens]
            if scheduler.waiting or pending:
                parked.extend(r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens)
                scheduler.running = pending
            else:
                scheduler.running.extend(parked)
                parked.clear()
                preparing = False
        before = {rid: (r.num_computed_tokens, r.num_prompt_tokens)
                  for rid, r in scheduler.requests.items()}
        output = original_schedule(*args, **kwargs)
        prefill, decode = 0, 0
        for rid, count in output.num_scheduled_tokens.items():
            if before[rid][0] < before[rid][1]:
                prefill += count
            else:
                if count != 1:
                    raise RuntimeError("Expected one decode token per request")
                decode += count
        if prefill and decode:
            raise RuntimeError("Mixed step despite preparation barrier")
        phases.append(dict(prefill_tokens=prefill, decode_requests=decode,
                           preemptions=len(output.preempted_req_ids or [])))
        return output

    def step():
        before = replays
        start = time.perf_counter()
        result = original_step()
        steps.append(dict(ms=(time.perf_counter() - start) * 1000, replays=replays - before))
        return result

    scheduler.schedule, engine.step, graph_class.replay = schedule, step, replay
    trace = []
    try:
        yield trace
        if len(phases) != len(steps):
            raise RuntimeError("Scheduler and engine step traces differ")
        trace.extend(dict(phase, **timing) for phase, timing in zip(phases, steps))
    finally:
        scheduler.schedule, engine.step, graph_class.replay = original_schedule, original_step, original_replay
        scheduler.running.extend(parked)


def summarize(trace, batch, input_len, output_len):
    if not trace or any(s["preemptions"] for s in trace):
        raise ValueError("Missing trace or KV preemption")
    if any(not math.isfinite(s["ms"]) or s["ms"] <= 0 for s in trace):
        raise ValueError("Invalid step time")
    if sum(s["prefill_tokens"] for s in trace) != batch * input_len:
        raise ValueError("Incomplete prefill coverage")
    prefill = [s for s in trace if s["prefill_tokens"]]
    decode = [s for s in trace if s["decode_requests"]]
    if (len(decode) != output_len - 1 or trace != prefill + decode
            or any(s["decode_requests"] != batch or s["prefill_tokens"] for s in decode)):
        raise ValueError("Incomplete full-batch decode or mixed phase trace")
    prefill_ms = sum(s["ms"] for s in prefill)
    result = dict(prefill_step_ms=prefill_ms,
                  prefill_tokens_per_second=batch * input_len * 1000 / prefill_ms,
                  full_batch_decode_steps=len(decode), steps=trace)
    if decode:
        total = sum(s["ms"] for s in decode)
        result.update(mean_decode_step_ms=total / len(decode),
                      median_decode_step_ms=statistics.median(s["ms"] for s in decode),
                      decode_tokens_per_second=batch * len(decode) * 1000 / total,
                      decode_graph_replays=sum(s["replays"] for s in decode))
    return result


def request_ttft_ms(outputs):
    values = [o.metrics.first_token_latency if o.metrics is not None else None for o in outputs]
    if any(t is None or not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("Missing/invalid vLLM TTFT; enable log stats")
    return [t * 1000 for t in values]


def make_prompts(tokenizer, batch, length):
    body = tokenizer.encode("The observatory recorded temperature and rainfall every day. ",
                            add_special_tokens=False)
    prompts = []
    for index in range(batch):
        prefix = tokenizer.encode(f"Document {index}. Read these records.\n", add_special_tokens=False)
        suffix = tokenizer.encode("\nSummarize the measurements.\n", add_special_tokens=False)
        needed = length - len(prefix) - len(suffix)
        if needed < 0 or not body:
            raise ValueError("Input length is too short for the benchmark prompt")
        tokens = prefix + (body * ((needed + len(body) - 1) // len(body)))[:needed] + suffix
        prompts.append({"prompt_token_ids": tokens})
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--linear", choices=("stock", "residue", "mext"), default="stock")
    parser.add_argument("--attention", choices=("stock", "resattn"), default="stock")
    parser.add_argument("--kv", choices=("auto", "nvfp4"), default="nvfp4")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--kv-gib", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    args.output_len = args.output_len if args.output_len is not None else (1 if args.phase == "prefill" else 512)
    if min(args.batch, args.input_len, args.output_len, args.kv_gib,
           args.trials, args.max_num_batched_tokens) <= 0:
        parser.error("Lengths, batch, KV budget and trials must be positive")
    if (args.phase == "prefill" and args.output_len != 1
            or args.phase == "decode" and args.output_len < 2):
        parser.error("Prefill needs output length 1; decode needs at least 2")
    args.model = args.model.resolve(strict=True)
    quant = configure(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.with_suffix(".claim").open("x") as stream:
        stream.write(str(args.model))
    if args.result.exists():
        raise FileExistsError(args.result)
    assert_gpu_exclusive(0)
    before_gpu = gpu_state(0)
    import torch
    import vllm
    import flashinfer
    from vllm import LLM, SamplingParams
    if args.linear != "stock":
        from quant.vllm_integration.patch import patch_vllm_modelopt
        patch_vllm_modelopt(str(args.model))
    if args.attention == "resattn":
        from quant.vllm_integration.attn.register import install
        if not install(force=True):
            raise RuntimeError("ResAttn registration failed")
    kwargs = dict(model=str(args.model), dtype="bfloat16", tensor_parallel_size=1,
                  kv_cache_dtype=args.kv, calculate_kv_scales=False, block_size=128,
                  max_model_len=args.input_len + args.output_len,
                  max_num_seqs=args.batch, max_num_batched_tokens=args.max_num_batched_tokens,
                  generation_config="vllm", disable_log_stats=False, kv_cache_memory_bytes=args.kv_gib * 1024**3,
                  enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
                  enforce_eager=args.enforce_eager,
                  attention_backend="CUSTOM" if args.attention == "resattn" else "FLASHINFER", seed=17,
                  kernel_config={"enable_flashinfer_autotune": True},
                  compilation_config={"pass_config": {"fuse_attn_quant": False},
                      "max_cudagraph_capture_size": args.batch,
                      "compile_ranges_endpoints": [args.max_num_batched_tokens],
                      "inductor_compile_config": {"deterministic": True, "benchmark_combo_kernel": False}})
    started = time.perf_counter()
    llm = LLM(**kwargs)
    startup = time.perf_counter() - started
    args.out, args.arm = str(args.result.parent), args.result.stem
    write_effective_engine_config(llm, args)
    context = llm.llm_engine.vllm_config.compilation_config.static_forward_context
    impls = Counter(type(layer.impl).__name__ for layer in context.values() if hasattr(layer, "impl"))
    initial_scales = {name: {key: float(getattr(layer, key)) for key in
                            ("_q_scale_float", "_k_scale_float", "_v_scale_float") if hasattr(layer, key)}
                      for name, layer in context.items() if hasattr(layer, "_k_scale_float")}
    model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
    methods = Counter(type(layer.quant_method).__name__ for layer in model.modules()
                      if getattr(layer, "quant_method", None) is not None)
    if any("ResidueNvFp4" in name for name in methods) != (args.linear != "stock"):
        raise RuntimeError(f"Linear implementation mismatch: {methods}")
    if any("ResAttn" in name for name in impls) != (args.attention == "resattn"):
        raise RuntimeError(f"Attention implementation mismatch: {impls}")
    tokenizer = llm.get_tokenizer()
    prompts = make_prompts(tokenizer, args.batch, args.input_len)
    sampling = dict(temperature=0, top_p=1, top_k=-1, min_p=0, repetition_penalty=1,
                    presence_penalty=0, frequency_penalty=0, seed=17)
    records = []
    params = SamplingParams(**sampling, max_tokens=args.output_len, ignore_eos=True)
    for trial in range(args.trials + 1):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with phase_trace(llm.llm_engine, torch.cuda.CUDAGraph) as trace:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if len(outputs) != args.batch or any(len(o.outputs) != 1 or
                len(o.outputs[0].token_ids) != args.output_len for o in outputs):
            raise ValueError("Incomplete generation")
        rec = summarize(trace, args.batch, args.input_len, args.output_len)
        rec.update(trial=trial, generation_ms=elapsed, ttft_ms=request_ttft_ms(outputs),
                   output_tokens_per_second=args.batch * args.output_len * 1000 / elapsed,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   token_sha256=hashlib.sha256(json.dumps([o.outputs[0].token_ids for o in outputs]).encode()).hexdigest())
        records.append(rec)
        print("TRIAL", json.dumps({k: v for k, v in rec.items() if k != "steps"}), flush=True)
        args.result.with_suffix(".partial.json").write_text(json.dumps(records, indent=2))
    report = dict(status="PASS", args=vars(args), quantization=quant, engine_kwargs=kwargs,
                  startup_seconds=startup, attention_impls=dict(impls), linear_methods=dict(methods), scales=initial_scales,
                  gpu_before=before_gpu, gpu_after=gpu_state(0),
                  runtime={"torch": torch.__version__, "vllm": vllm.__version__,
                           "flashinfer": flashinfer.__version__, "cuda": torch.version.cuda},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                  prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                  runtime_env={k: v for k, v in os.environ.items() if k.startswith(("RESIDUE_", "VLLM_"))},
                  warmup=records[0], trials=records[1:],
                  protocol="Prepared real KV; synchronous engine-step wall time, not CUDA kernel time; first output token belongs to prefill")
    args.result.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
