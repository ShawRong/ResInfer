"""Compare real decode and optional prefill operator outputs against FP32 references.

Run with python -m benchmarks.precision.run --help.
"""

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

from .compare import (LINEAR_ARMS, LINEAR_SCOPE, check_linear_coverage,
                      compare_attention, compare_linear, compare_prefill, summarize)
from .prepare import load_protocol, load_kext_metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Local BF16 checkpoint"
    )
    parser.add_argument(
        "--protocol", type=Path, required=True, help="Frozen replay JSON"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory"
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        help="Save first-step tensors, including a full linear weight",
    )
    parser.add_argument(
        "--timing", action="store_true", help="Also record isolated linear Graph timing"
    )
    parser.add_argument("--kext-metadata", action="append", default=[], metavar="RATIO=PATH",
                        help="Capture prefill and compare KEXT using calibrated export masks; repeat per ratio")
    args = parser.parse_args()
    if args.max_model_len < 2:
        parser.error("--max-model-len must be at least 2")
    return args


def load_model(model_path, max_model_len):
    from vllm import LLM

    from benchmarks.engine_config import accuracy_engine_config

    llm = LLM(
        **accuracy_engine_config(
            {
                "model": model_path,
                "tokenizer": model_path,
                "dtype": "bfloat16",
                "quantization": None,
                "tensor_parallel_size": 1,
                "enforce_eager": True,
                "kv_cache_dtype": "nvfp4",
                "calculate_kv_scales": False,
                "block_size": 128,
                "max_model_len": max_model_len,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 4096,
                "kv_cache_memory_bytes": 4 * 1024**3,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "async_scheduling": False,
                "attention_backend": "CUSTOM",
                "seed": 42,
                "kernel_config": {"enable_flashinfer_autotune": False},
                "compilation_config": {"pass_config": {"fuse_attn_quant": False}},
            },
            {},
        )
    )
    return llm


def prefill_span(counts, layer, prompt_tokens, rows):
    """B1 eager replay visits prompt tokens once, possibly in multiple chunks."""
    start = counts.get(layer, 0)
    count = min(rows, prompt_tokens - start)
    counts[layer] = start + count
    return start, count


def collect_inputs(llm, protocol, root, save_inputs, rows, costs, save, prefill_inputs=None, kext_metadata=None):
    """Replay tokens, capture decode/prefill inputs and restore all hooks."""
    import torch
    from vllm import SamplingParams
    from vllm.model_executor.layers.linear import LinearBase

    from quant.vllm_integration.attn.backend import ResAttnFlashInferImpl

    case = None
    generated = 0
    names, history, activations, weights = {}, {}, {}, {}
    hooks = []
    prefill_counts = {}
    original = ResAttnFlashInferImpl.forward

    @torch.inference_mode()
    def observed(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        md,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        name = names.get(id(layer))
        if case is None or name is None or md is None:
            return original(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                md,
                output,
                output_scale,
                output_block_scale,
            )
        assert output_scale is None and output_block_scale is None
        if md.num_prefill_tokens:
            n = md.num_actual_tokens
            history[name][0].append(key[:n].detach().clone())
            history[name][1].append(value[:n].detach().clone())
        else:
            assert md.num_decodes == 1
            history[name][0].append(key[:1].detach().clone())
            history[name][1].append(value[:1].detach().clone())
        result = original(self, layer, query, key, value, kv_cache, md, output)
        if md.num_prefill_tokens or generated not in protocol["steps"]:
            return result
        compared, cost = compare_attention(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            md,
            output,
            history[name],
            case,
            generated,
            name,
            root,
            save_inputs,
        )
        rows.extend(compared)
        costs.append(cost)
        save("collecting")
        print("ATTN", case["index"], generated, name, flush=True)
        return result

    model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
    original_logits = model.compute_logits
    try:
        for name, module in model.named_modules():
            if not any(f"layers.{i}." in name for i in protocol["layers"]):
                continue
            if isinstance(getattr(module, "impl", None), ResAttnFlashInferImpl):
                names[id(module)] = name
            if isinstance(module, LinearBase) and name.rsplit(".", 1)[-1] in [
                "qkv_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
            ]:
                assert (
                    module.weight.dtype == torch.bfloat16
                    and getattr(module, "bias", None) is None
                )
                weights[name] = module.weight.detach()
                if kext_metadata:
                    from benchmarks.kernel.inputs import calibrated_indices
                    for ratio, data in kext_metadata.items():
                        calibrated_indices(data, name, module.weight.shape[1], ratio)

                def capture(mod, args, label=name):
                    if case is not None:
                        start, count = prefill_span(prefill_counts, label, len(case["prompt_ids"]),
                                                    args[0].shape[0])
                        if count and prefill_inputs is not None:
                            prefill_inputs[(case["index"], start, label)] = (
                                args[0][:count].detach().to("cpu", copy=True))
                    if (
                        case is not None
                        and generated in protocol["steps"]
                        and args[0].shape[0] == 1
                    ):
                        activations[(case["index"], generated, label)] = (
                            args[0].detach().clone()
                        )

                hooks.append(module.register_forward_pre_hook(capture))
        assert len(names) == len(protocol["layers"]) and len(weights) == 4 * len(
            protocol["layers"]
        ), (names, list(weights))
        write_runtime(llm, root, names, weights)

        @torch.inference_mode()
        def force_logits(*a, **kw):
            nonlocal generated
            logits = original_logits(*a, **kw)
            # vLLM also computes discarded logits for intermediate prefill chunks.
            if case is not None and all(
                    prefill_counts.get(name, 0) == len(case["prompt_ids"]) for name in weights):
                assert logits.shape[0] == 1
                logits.fill_(-torch.inf)
                logits[0, case["tokens"][generated]] = 0
                generated += 1
            return logits

        model.compute_logits = force_logits
        ResAttnFlashInferImpl.forward = observed
        for selected in protocol["cases"]:
            case = selected
            generated = 0
            prefill_counts.clear()
            history = {name: ([], []) for name in names.values()}
            t = time.time()
            result = llm.generate(
                [{"prompt_token_ids": case["prompt_ids"]}],
                SamplingParams(
                    temperature=0,
                    seed=42,
                    max_tokens=len(case["tokens"]),
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )[0].outputs[0]
            assert list(result.token_ids) == case["tokens"]
            if prefill_inputs is not None and any(
                    prefill_counts.get(name) != len(case["prompt_ids"]) for name in weights):
                raise ValueError("Incomplete prefill activation capture")
            duration = time.time() - t
            costs.append(
                {
                    "stage": "diagnostic_generation",
                    "index": case["index"],
                    "tokens": len(result.token_ids),
                    "seconds": duration,
                    "tokens_per_second": len(result.token_ids) / duration,
                }
            )
            print("CASE", case["index"], duration, flush=True)
    finally:
        ResAttnFlashInferImpl.forward = original
        model.compute_logits = original_logits
        for hook in hooks:
            hook.remove()
    return weights, activations


def write_runtime(llm, root, names, weights):
    import importlib.metadata
    from importlib.util import find_spec

    import torch

    model_path = llm.llm_engine.vllm_config.model_config.model
    native_library_path = find_spec("nvfp4._C").origin
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "model": model_path,
                "config_sha256": hashlib.sha256(
                    (Path(model_path) / "config.json").read_bytes()
                ).hexdigest(),
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in [
                        "torch",
                        "vllm",
                        "flashinfer-python",
                        "nvidia-cutlass-dsl",
                    ]
                },
                "native_sha256": hashlib.sha256(
                    Path(native_library_path).read_bytes()
                ).hexdigest(),
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "source_root": str(Path(__file__).resolve().parents[2]),
                "mode": "eager fixed-token collection",
                "layers": list(names.values()),
                "weight_modules": list(weights),
                "tf32": False,
                "compilation": str(llm.llm_engine.vllm_config.compilation_config),
            },
            indent=2,
        )
    )


def configure_runtime():
    import os

    import torch

    os.environ.update(
        RESIDUE_NVFP4="0",
        RESIDUE_ATTN="1",
        RESIDUE_ATTN_CUDAGRAPH="1",
        RESIDUE_ATTN_REDUCTION="auto",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        RESIDUE_SM10X_FOLD_TUNER="1",
        RESIDUE_RATIO1_PREFILL_PATH="auto",
    )
    # Registers torch.ops.residue.nvfp4_linear used by the linear runners.
    from quant.vllm_integration import residue_linear_op  # noqa: F401
    from quant.vllm_integration.attn.register import install

    assert install(force=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_results(root, protocol, started, rows, costs, state):
    import torch

    data = {
        "status": state,
        "scope": protocol["scope"],
        "mext_scope": LINEAR_SCOPE,
        "linear_arms": LINEAR_ARMS,
        "kext_arms": sorted({r["arm"] for r in rows if r["kind"] == "kext"}),
        "elapsed": time.time() - started,
        "rows": rows,
        "summary": summarize(rows) if state == "completed" else [],
        "costs": costs,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "frozen_sha256": hashlib.sha256(
            (root / "frozen.json").read_bytes()
        ).hexdigest(),
        "reference_impl": "torch_sdpa_math+vllm_fp32_kv_c9e62f13",
    }
    p = root / "results.part"
    p.write_text(json.dumps(data, indent=2))
    p.replace(root / "results.json")


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    metadata, sources = load_kext_metadata(args.kext_metadata)
    model_path = str(args.model.resolve(strict=True))
    if any(
        len(c["prompt_ids"]) + len(c["tokens"]) > args.max_model_len
        for c in protocol["cases"]
    ):
        raise ValueError("Frozen replay exceeds --max-model-len")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if metadata:
        for ratio, data in metadata.items():
            (root / f"kext-r{ratio:g}.json").write_text(json.dumps(data, indent=2))
        (root / "kext-sources.json").write_text(json.dumps(sources, indent=2))
    configure_runtime()
    (root / "frozen.json").write_bytes(args.protocol.read_bytes())
    started = time.time()
    rows, costs = [], []
    prefill_inputs = {} if metadata else None

    save = partial(write_results, root, protocol, started, rows, costs)

    try:
        llm = load_model(model_path, args.max_model_len)
        costs.append({"stage": "startup", "seconds": time.time() - started})
        weights, activations = collect_inputs(
            llm, protocol, root, args.save_inputs, rows, costs, save, prefill_inputs, metadata
        )
        compare_linear(
            weights,
            activations,
            protocol,
            root,
            args.save_inputs,
            args.timing,
            rows,
            costs,
            save,
        )
        if metadata:
            compare_prefill(weights, prefill_inputs, metadata, args.timing, rows, costs, save)
        expected = len(protocol["cases"]) * len(protocol["steps"])
        assert (
            sum(r["kind"] == "resattn" for r in rows)
            == expected * len(protocol["layers"]) * 2
        )
        check_linear_coverage(rows, protocol, weights)
        save("completed")
        print("COMPLETED", len(rows), flush=True)
    finally:
        from torch import distributed

        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
