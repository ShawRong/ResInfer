"""Bounded serving-startup and phase-separated decode gate for ResAttn tuning."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=8)
    args = parser.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from quant.vllm_integration.attn.decode import ResAttnDecodeMixin

    replays = Counter()
    replay = torch.cuda.CUDAGraph.replay
    tuned_decode = ResAttnDecodeMixin._try_tuned_decode
    captured_calls = []

    def tracked_decode(impl, *a, **kw):
        result = tuned_decode(impl, *a, **kw)
        if torch.cuda.is_current_stream_capturing() and result:
            captured_calls.append(impl._decode_tuning_status)
        return result

    def tracked(graph):
        caller = inspect.currentframe().f_back.f_code.co_filename
        replays[caller] += 1
        return replay(graph)

    torch.cuda.CUDAGraph.replay = tracked
    ResAttnDecodeMixin._try_tuned_decode = tracked_decode
    try:
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=1,
            kv_cache_dtype="nvfp4", block_size=128,
            max_model_len=args.seq + args.output_len,
            max_num_seqs=args.batch, max_num_batched_tokens=args.batch * args.seq,
            kv_cache_memory_bytes=2 * 1024**3,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, attention_backend="CUSTOM", seed=17,
            kernel_config={"enable_flashinfer_autotune": True},
            compilation_config={"cudagraph_capture_sizes": [1, args.batch],
                                "max_cudagraph_capture_size": args.batch,
                                "compile_ranges_endpoints": [2048]},
        )
        reports = llm.collective_rpc(
            lambda worker: worker.model_runner._resattn_tuning_report)
        assert reports and all(reports)
        captured = llm.collective_rpc(lambda worker: [
            dict(status=layer.impl._decode_tuning_status,
                 workspaces=len(layer.impl._decode_reduction_workspaces))
            for layer in worker.model_runner.compilation_config.static_forward_context.values()
            if hasattr(getattr(layer, "impl", None), "_decode_tuning_status")])
        assert captured and all(captured)
        assert all(row["status"][1] in ("autotuned", "autotuned_fallback")
                   for rank in captured for row in rank), captured
        assert {row[2] for row in captured_calls} == {1, args.batch}, captured_calls
        assert all(row[1] in ("autotuned", "autotuned_fallback")
                   for row in captured_calls), captured_calls
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        schedule = scheduler.schedule
        phases = []

        def checked(*a, **kw):
            before = {rid: (req.num_computed_tokens, req.num_prompt_tokens)
                      for rid, req in scheduler.requests.items()}
            result = schedule(*a, **kw)
            prefill = decode = 0
            for rid, count in result.num_scheduled_tokens.items():
                if before[rid][0] < before[rid][1]:
                    prefill += count
                else:
                    assert count == 1
                    decode += count
            assert not (prefill and decode), "mixed batch"
            phases.append((prefill, decode))
            return result

        scheduler.schedule = checked
        replays.clear()
        outputs = llm.generate(
            [{"prompt_token_ids": [1, 5, 6, 7] * (args.seq // 4)
              + [8] * (args.seq % 4)} for _ in range(args.batch)],
            SamplingParams(temperature=0, max_tokens=args.output_len,
                           ignore_eos=True, detokenize=False, logprobs=1),
            use_tqdm=False)
        assert phases == [(args.batch * args.seq, 0)] + [(0, args.batch)] * (args.output_len - 1)
        assert len(outputs) == args.batch
        tokens = []
        for out in outputs:
            sequence = out.outputs[0]
            assert len(sequence.token_ids) == args.output_len
            assert all(math.isfinite(step[token].logprob)
                       for token, step in zip(sequence.token_ids, sequence.logprobs))
            tokens.append(list(sequence.token_ids))
        model_replays = sum(count for path, count in replays.items()
                            if "compilation/cuda_graph.py" in path or "cudagraph_utils.py" in path)
        assert model_replays > 0, "model CUDA Graph was not replayed"
        result = dict(status="passed", startup=reports, captured=captured,
                      captured_calls=captured_calls, phases=phases,
                      model_graph_replays=model_replays,
                      token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                      tokens=tokens)
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        torch.cuda.CUDAGraph.replay = replay
        ResAttnDecodeMixin._try_tuned_decode = tuned_decode


if __name__ == "__main__":
    main()
