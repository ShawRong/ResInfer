"""Matched no-AWQ NVFP4/MXFP4 plain / Residue export from local HF and JSONL inputs.

Run as ``python -m export.export_nvfp4 --help``. Outputs are create-only.
Preflight measures BF16 calibration forward throughput; export records the
actual quantization forward timings separately. Runtime acceptance is separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_rows(path, samples, seed):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if samples <= 0 or samples > len(rows):
        raise ValueError(f"requested {samples} rows, available {len(rows)}")
    indices = random.Random(seed).sample(range(len(rows)), samples)
    return [(i, rows[i]) for i in indices]


def render_row(row, tokenizer, reasoning_effort="medium"):
    if isinstance(row.get("messages"), list) and row["messages"]:
        messages = copy.deepcopy(row["messages"])
        for message in messages:
            content = message.get("content")
            if message.get("role") != "assistant" or not isinstance(content, str):
                continue
            content = content.lstrip()
            if not content.startswith("<think>"):
                continue
            if content.count("<think>") != 1 or content.count("</think>") > 1:
                raise ValueError("ambiguous embedded assistant thinking in calibration data")
            reasoning, separator, answer = content[len("<think>"):].partition("</think>")
            if not separator:
                raise ValueError("unterminated embedded assistant thinking in calibration data")
            existing = message.get("reasoning_content")
            if existing and existing.strip() != reasoning.strip():
                raise ValueError("conflicting embedded and structured reasoning_content")
            message["reasoning_content"] = reasoning
            message["content"] = answer.lstrip()
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False,
                                             enable_thinking=True,
                                             preserve_thinking=True,
                                             reasoning_effort=reasoning_effort)
        for message in messages:
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip() and reasoning.strip() not in rendered:
                raise ValueError("chat template dropped calibration reasoning_content")
        return rendered
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    raise ValueError("calibration row requires nonempty text or messages")



def validate_modelopt_registry():
    """Fail before loading weights if optional integrations corrupt the registry."""
    import modelopt.torch.quantization  # register the supported module classes
    import quant.modelopt_integration  # register Residue algorithms
    from modelopt.torch.quantization.nn import QuantModuleRegistry
    bad = [getattr(cls, "__qualname__", repr(cls))
           for cls in QuantModuleRegistry._key_registry if not isinstance(cls, type)]
    if bad:
        raise RuntimeError(
            f"ModelOpt registered non-class modules: {bad}. Use a dedicated HF "
            "export environment without incompatible optional runtime plugins."
        )


def load_model(source, device):
    import torch
    from transformers import AutoConfig
    from export.stages.hf_model_class import resolve_hf_auto_model_class
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    loader = resolve_hf_auto_model_class(config)
    model, info = loader.from_pretrained(
        source, dtype=torch.bfloat16, device_map=device,
        local_files_only=True, output_loading_info=True,
    )
    # HF does not instantiate Qwen MTP. The common exporter restores precisely
    # these source tensors through extra_state_dict and checks their hashes.
    unexpected = [k for k in info.get("unexpected_keys", [])
                  if not (config.model_type == "qwen3_5" and k.startswith("mtp."))]
    bad = {k: info[k] for k in ("missing_keys", "mismatched_keys", "error_msgs") if info.get(k)}
    if unexpected:
        bad["unexpected_keys"] = unexpected
    if bad:
        raise RuntimeError(f"unclean checkpoint load: {bad}")
    model.eval()
    model.config.use_cache = False
    return model, info


def quant_config(model, residue, ratio=0.125, fmt="nvfp4"):
    import torch
    import modelopt.torch.quantization as mtq
    from quant.modelopt_integration.registry import build_residue_cfg
    cfg = (build_residue_cfg(fmt, ratio, calib_block_size=8)
           if residue else copy.deepcopy(
               mtq.NVFP4_DEFAULT_CFG if fmt == "nvfp4" else mtq.MXFP4_DEFAULT_CFG))
    # Explicit allowlist: all language-trunk linears, excluding small GDN gates.
    # Never calibrate an unvisited vision module from text-only calibration.
    prefix = "model.language_model.layers." if model.config.model_type == "qwen3_5" else "model.layers."
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
             and n.startswith(prefix) and not n.endswith(("in_proj_a", "in_proj_b"))]
    if not names:
        raise ValueError("no language linear modules selected")
    qc = cfg["quant_cfg"]
    if isinstance(qc, list):
        qc.append({"quantizer_name": "*", "enable": False})
        for n in names:
            for suffix in ("input_quantizer", "weight_quantizer"):
                qc.append({"quantizer_name": n + "." + suffix, "enable": True})
    else:
        qc["*"] = {"enable": False}
        for n in names:
            for suffix in ("input_quantizer", "weight_quantizer"):
                qc[n + "." + suffix] = {"enable": True}
    return cfg, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=("preflight", "plain", "residue"), required=True)
    ap.add_argument("--format", choices=("nvfp4", "mxfp4"), default="nvfp4")
    ap.add_argument("--ratio", type=float, choices=(0.125, 0.25, 0.5, 1.0), default=0.125)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reasoning-effort", choices=("medium", "xhigh", "low"), default="medium")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.seq_len, args.batch_size) < 1:
        raise ValueError("sequence length and batch size must be positive")
    import torch
    import torch._inductor.config as ic
    from transformers import AutoTokenizer
    ic.deterministic = True
    ic.benchmark_combo_kernel = False
    torch.manual_seed(args.seed)
    validate_modelopt_registry()
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    original_padding = tokenizer.pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = select_rows(args.dataset, args.samples, args.seed)
    encoded = [tokenizer(render_row(row, tokenizer, args.reasoning_effort), truncation=True,
                         max_length=args.seq_len, add_special_tokens=False)["input_ids"]
               for _, row in rows]
    if any(not ids for ids in encoded):
        raise ValueError("empty tokenized calibration sample")
    args.output.mkdir(parents=True)
    provenance = dict(dataset_sha256=sha(args.dataset), source=str(args.source),
                      source_index_sha256=sha(args.source / "model.safetensors.index.json"),
                      samples=args.samples, seq_len=args.seq_len, batch_size=args.batch_size,
                      seed=args.seed, mode=args.mode, awq=False,
                      quant_format=args.format, residue_ratio=args.ratio if args.mode == "residue" else 0,
                      salience_statistic="batch_mean_abs" if args.mode == "residue" else None,
                      rendering_policy="assistant_embedded_think_to_reasoning_content_v1",
                      template_kwargs=dict(enable_thinking=True, preserve_thinking=True,
                                           reasoning_effort=args.reasoning_effort,
                                           add_generation_prompt=False),
                      row_indices=[i for i, _ in rows], token_lengths=[len(x) for x in encoded],
                      token_ids_sha256=hashlib.sha256(json.dumps(encoded).encode()).hexdigest())
    (args.output / "recipe.json").write_text(json.dumps(provenance, indent=2))
    model, info = load_model(args.source, args.device)
    (args.output / "load-report.json").write_text(json.dumps(info, indent=2, default=str))
    timings = []

    def forward(active_model, batch_size=args.batch_size, limit=None):
        selected = encoded if limit is None else encoded[:limit]
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(selected), batch_size):
                ids = selected[start:start + batch_size]
                batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(args.device)
                result = active_model(**batch, use_cache=False, logits_to_keep=1)
                if not result.logits.isfinite().all().item():
                    raise ValueError(f"nonfinite calibration forward at row {start}")
                del result
                if (start // batch_size + 1) % 8 == 0 or start + batch_size >= len(selected):
                    progress = dict(completed_samples=min(start + batch_size, len(selected)),
                                    total_samples=len(selected), seconds=time.monotonic()-started)
                    (args.output / "calibration-progress.json").write_text(json.dumps(progress))
                    print("calibration_progress", json.dumps(progress), flush=True)
        torch.cuda.synchronize()
        record = dict(batch_size=batch_size, samples=len(selected),
                      tokens=sum(map(len, selected)), seconds=time.monotonic()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        timings.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "timings.json").write_text(json.dumps(timings, indent=2))

    if args.mode == "preflight":
        forward(model, 1, 1)
        for batch_size in (1, 4, 8):
            torch.cuda.reset_peak_memory_stats()
            forward(model, batch_size, min(16, len(encoded)))
        (args.output / "complete.json").write_text(json.dumps({"status": "preflight_passed"}))
        return
    import modelopt.torch.quantization as mtq
    from quant.modelopt_integration import inject_residue_salient_map
    from quant.modelopt_integration.export import export_residue_checkpoint
    cfg, names = quant_config(model, args.mode == "residue", args.ratio, args.format)
    (args.output / "quant-config.json").write_text(json.dumps(cfg, indent=2, default=str))
    torch.cuda.reset_peak_memory_stats()
    model = mtq.quantize(model, cfg, forward_loop=forward)
    modules = dict(model.named_modules())
    enabled = {n for n, m in modules.items()
               if getattr(getattr(m, "weight_quantizer", None), "is_enabled", False)}
    if enabled != set(names):
        raise ValueError(f"quantization scope mismatch: missing={set(names)-enabled}, extra={enabled-set(names)}")
    if args.format == "nvfp4":
        for n in names:
            for suffix in ("input_quantizer", "weight_quantizer"):
                q = getattr(modules[n], suffix)
                if q.amax is None or not q.amax.isfinite().all().item():
                    raise ValueError(f"uncalibrated quantizer: {n}.{suffix}")
    if args.mode == "residue":
        from quant.modelopt_integration.fused_salience import align_runtime_fused_salience
        alignment = align_runtime_fused_salience(model, ratio=args.ratio, block_size=8,
                                                 salience_metric="activation_max")
        (args.output / "fused-salience.json").write_text(json.dumps(alignment, indent=2))
        salient = {n: modules[n].input_quantizer._salient_indices for n in names}
        if any(v is None or not v.numel() for v in salient.values()):
            raise ValueError("missing residue channel selection")
        inject_residue_salient_map(model, salient, residue_ratio=args.ratio,
                                  block_size=8)
    tokenizer.pad_token = original_padding
    # Preserve original generation/cache configuration; calibration disabled KV only.
    from transformers import AutoConfig
    source_config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    model.config.use_cache = getattr(source_config, "use_cache", True)
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = getattr(source_config.text_config, "use_cache", True)
    output = args.output / "checkpoint"
    export_residue_checkpoint(model, output, dtype=torch.bfloat16,
                              source_checkpoint=args.source, tokenizer=tokenizer,
                              residue_ratio=args.ratio, calibration=provenance,
                              save_kernel_metadata=args.mode == "residue", max_shard_size="4GB")
    from export.hf_runtime_assets import copy_hf_runtime_assets
    copy_hf_runtime_assets(args.source, output)
    # Template/tokenizer/generation assets must be identical to the fine-tune.
    import shutil
    for name in ("generation_config.json", "tokenizer_config.json", "tokenizer.json",
                 "chat_template.jinja", "vocab.json", "merges.txt"):
        if (args.source / name).is_file():
            shutil.copy2(args.source / name, output / name)
    from export.stages.inspect_quant_export import InspectQuantExportStage
    if args.mode == "residue":
        meta = json.loads((output / "residue_kernel_metadata.json").read_text())
        if args.format == "mxfp4":
            InspectQuantExportStage._assert_mxfp4_tensors(output, meta["layers"])
        else:
            InspectQuantExportStage._assert_extended_widths(output, meta["layers"])
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata_file_sha256"] = {p.name: sha(p) for p in output.iterdir()
                                        if p.is_file() and p.suffix != ".safetensors" and p != manifest_path}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (args.output / "complete.json").write_text(json.dumps({"status": "structure_validated",
                                                         "runtime_validated": False}))


if __name__ == "__main__":
    main()
