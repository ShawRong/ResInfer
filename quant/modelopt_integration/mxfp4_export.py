# SPDX-License-Identifier: Apache-2.0
"""ModelOpt 0.45 native MXFP4 export -> explicit SM100 ResInfer checkpoint.

Packing and zero-block padding remain ModelOpt's responsibility. This adapter
describes residue/fusion contracts and validates the official export bytes.
The initial runtime target is dense Llama, TP1. MoE export is rejected.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch

from quant.mxfp4_checkpoint import FORMAT, runtime_layer_specs, padded_k, validate_layer


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(tensor):
    array = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def _require_modelopt_export_api(export_hf_checkpoint):
    version = importlib.metadata.version("nvidia-modelopt")
    if version.split(".")[:2] != ["0", "45"]:
        raise ValueError(
            f"MXFP4 export is validated with ModelOpt 0.45.x, got {version}"
        )
    required = {"model", "dtype", "export_dir", "save_modelopt_state",
                "extra_state_dict", "max_shard_size"}
    missing = required - set(inspect.signature(export_hf_checkpoint).parameters)
    if missing:
        raise ValueError(f"ModelOpt exporter is missing required parameters: {sorted(missing)}")
    return version


def has_mxfp4_weights(model):
    return any(getattr(q, "is_enabled", False)
               and getattr(q, "num_bits", None) == (2, 1)
               and (getattr(q, "block_sizes", None) or {}).get("scale_bits") == (8, 0)
               for module in model.modules() for q in [getattr(module, "weight_quantizer", None)])


def collect_mxfp4_layers(model):
    from modelopt.torch.export.quant_utils import get_quantization_format
    layers = {}
    for name, module in model.named_modules():
        wq = getattr(module, "weight_quantizer", None)
        if wq is None or not wq.is_enabled:
            continue
        if not isinstance(module, torch.nn.Linear) or get_quantization_format(module) != "mxfp4":
            raise ValueError(f"only dense MXFP4 W4A4 is supported: {name}")
        if any(getattr(q, "pre_quant_scale", None) is not None
               for q in (wq, module.input_quantizer)):
            raise ValueError(f"fuse pre_quant_scale before MXFP4 export: {name}")
        indices = getattr(module.input_quantizer, "salient_indices", None)
        mode = getattr(module, "_residue_runtime_mode", None)
        if mode == "mext_r1":
            k = module.weight.shape[1]
            if indices is None or indices.tolist() != list(range(k)):
                raise ValueError(f"full residue requires all channels in order: {name}")
            indices = None
        elif indices is not None and indices.numel():
            mode = "selective"
            k = getattr(module, "_original_in_features", None)
            expected = padded_k(int(k), int(indices.numel())) if k is not None else None
            if k is None or module.weight.shape[1] != expected:
                raise ValueError(
                    f"materialize and pad selected weights before export: {name}; "
                    f"expected K={expected}, got {module.weight.shape[1]}"
                )
        else:
            mode, k = "standard", module.weight.shape[1]
        spec = dict(mode=mode, input_size=int(k), output_size=module.weight.shape[0])
        if mode == "selective":
            spec["indices"] = indices.detach().cpu().tolist()
        validate_layer(spec)
        if not torch.isfinite(module.weight).all().item():
            raise ValueError(f"nonfinite source weight: {name}")
        layers[name] = spec
    if not layers:
        raise ValueError("no enabled MXFP4 linear layers found")
    return layers


def _validate_modelopt_export(path, hf_layers, preserved):
    """Validate official ModelOpt output and remove only private salience buffers."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    native_config = config.get("quantization_config")
    if (not isinstance(native_config, dict)
            or native_config.get("quant_method") != "modelopt"
            or str(native_config.get("quant_algo", "")).lower() != "mxfp4"):
        raise ValueError(f"unexpected ModelOpt MXFP4 config: {native_config!r}")
    seen, hashes, weight_map, total_size = set(), {}, {}, 0
    removed_keys = set()
    expected_quant = {
        f"{name}.{key}" for name in hf_layers for key in ("weight", "weight_scale")
    }
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise ValueError("ModelOpt export produced no safetensors")
    index_path = path / "model.safetensors.index.json"
    native_index = json.loads(index_path.read_text()) if index_path.exists() else None
    native_weight_map = (native_index or {}).get("weight_map", {})

    for shard in shards:
        tensors, shard_removed = {}, False
        with safe_open(shard, framework="pt") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"duplicate exported tensor: {key}")
                seen.add(key)
                if "quantizer." in key:
                    if key.endswith("._salient_indices"):
                        removed_keys.add(key)
                        shard_removed = True
                        continue
                    raise ValueError(f"unexpected calibration tensor: {key}")
                tensor = handle.get_tensor(key)
                prefix, leaf = key.rsplit(".", 1)
                if prefix in hf_layers and leaf in ("weight", "weight_scale"):
                    spec = hf_layers[prefix]
                    k = padded_k(spec["input_size"], len(spec.get("indices", [])))
                    width = k // (2 if leaf == "weight" else 32)
                    expected_shape = [spec["output_size"], width]
                    if tensor.dtype != torch.uint8 or list(tensor.shape) != expected_shape:
                        raise ValueError(
                            f"unexpected native ModelOpt MXFP4 shape/dtype: "
                            f"{key} {tensor.shape} {tensor.dtype}"
                        )
                    if leaf == "weight_scale" and (tensor == 255).any().item():
                        raise ValueError(f"NaN E8M0 code in {key}")
                    hashes[key] = _hash(tensor)
                elif prefix in hf_layers and leaf != "bias":
                    raise ValueError(f"unexpected MXFP4 auxiliary tensor: {key}")
                if tensor.is_floating_point() and not tensor.isfinite().all().item():
                    raise ValueError(f"nonfinite exported tensor: {key}")
                if key in preserved and _hash(tensor) != preserved[key]:
                    raise ValueError(f"unquantized parameter changed during export: {key}")
                tensors[key] = tensor.contiguous()
                weight_map[key] = shard.name
                total_size += tensor.numel() * tensor.element_size()
        if shard_removed:
            temporary = shard.with_suffix(".safetensors.tmp")
            save_file(tensors, temporary, metadata=metadata)
            temporary.replace(shard)

    if native_index is not None:
        filtered_native_map = {
            key: value for key, value in native_weight_map.items()
            if key not in removed_keys
        }
        if filtered_native_map != weight_map:
            raise ValueError("ModelOpt shard index does not match exported tensors")
    elif len(shards) != 1:
        raise ValueError("multi-shard ModelOpt export is missing its safetensors index")
    missing = (expected_quant | set(preserved)) - set(weight_map)
    if missing:
        raise ValueError(f"missing exported tensors: {sorted(missing)}")
    if native_index is not None and removed_keys:
        native_index["weight_map"] = weight_map
        native_index.setdefault("metadata", {})["total_size"] = total_size
        index_path.write_text(json.dumps(native_index, indent=2) + "\n")
    return config, native_config, hashes


def _collect_preserved_state(model, source_checkpoint, hf_layers,
                             preserve_unloaded_prefixes):
    """Return ModelOpt extra_state_dict plus hashes for unquantized tensors."""
    if source_checkpoint is None:
        hashes = {
            name: _hash(parameter)
            for name, parameter in model.named_parameters()
            if (name.rsplit(".", 1)[0] not in hf_layers or name.endswith(".bias"))
            and "quantizer" not in name
        }
        return {}, hashes, None

    from safetensors import safe_open

    source = Path(source_checkpoint).resolve()
    if not source.is_dir():
        raise ValueError(f"source checkpoint is not a directory: {source}")
    source_index = source / "model.safetensors.index.json"
    if source_index.is_file():
        index = json.loads(source_index.read_text())
        source_map = index.get("weight_map")
        if not isinstance(source_map, dict) or not source_map:
            raise ValueError(f"invalid source safetensors index: {source_index}")
        inventory = {
            "kind": "sharded_safetensors",
            "index_sha256": _file_hash(source_index),
        }
    else:
        source_file = source / "model.safetensors"
        if not source_file.is_file():
            raise ValueError(
                "source checkpoint must contain model.safetensors or "
                "model.safetensors.index.json"
            )
        with safe_open(source_file, framework="pt") as handle:
            source_map = {name: source_file.name for name in handle.keys()}
        inventory = {"kind": "single_safetensors", "filename": source_file.name}
    if any(not str(filename).endswith(".safetensors")
           for filename in source_map.values()):
        raise ValueError("source checkpoint index must reference safetensors shards")
    loaded_keys = set(model.state_dict())
    excluded = set(source_map) - loaded_keys
    if any(not name.startswith(tuple(preserve_unloaded_prefixes)) for name in excluded):
        raise ValueError(f"unloaded source tensor families: {sorted(excluded)[:10]}")

    extra_tensors, hashes = {}, {}
    for filename in sorted(set(source_map.values())):
        with safe_open(source / filename, framework="pt") as handle:
            for name in handle.keys():
                if name not in source_map or source_map[name] != filename:
                    raise ValueError(f"invalid source shard index: {name}")
                if name.endswith(".weight") and name[:-7] in hf_layers:
                    continue
                tensor = handle.get_tensor(name)
                hashes[name] = _hash(tensor)
                if name in excluded:
                    extra_tensors[name] = tensor
    expected = set(source_map) - {name + ".weight" for name in hf_layers}
    if expected != set(hashes):
        raise ValueError("source preservation inventory mismatch")
    record = {
        "checkpoint_name": source.name,
        "weight_inventory": inventory,
        "config_sha256": _file_hash(source / "config.json"),
        "source_tensor_count": len(source_map),
        "preserved_unloaded_tensors": sorted(excluded),
    }
    return extra_tensors, hashes, record


def export_mxfp4_checkpoint(model, export_dir, *, dtype=None, tokenizer=None,
                            calibration=None, max_shard_size="10GB", source_checkpoint=None,
                            processor=None, preserve_unloaded_prefixes=None):
    """Export to a new/empty directory, retaining native packed tensor hashes."""
    from modelopt.torch.export import export_hf_checkpoint
    _require_modelopt_export_api(export_hf_checkpoint)
    model_type = getattr(model.config, "model_type", None)
    if model_type not in ("llama", "qwen3_5"):
        raise ValueError("MXFP4 packed export supports dense Llama and Qwen3.5 VL")
    if model_type == "qwen3_5" and source_checkpoint is None:
        raise ValueError("Qwen3.5 requires the source checkpoint to preserve vision/MTP")
    if preserve_unloaded_prefixes is None:
        preserve_unloaded_prefixes = ("mtp.",) if model_type == "qwen3_5" else ()
    model_parameter = next(model.parameters())
    model_device = model_parameter.device
    path = Path(export_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite export: {path}")
    hf_layers = collect_mxfp4_layers(model)
    runtime_layers = runtime_layer_specs(hf_layers, model_type)
    extra_tensors, preserved, source_record = _collect_preserved_state(
        model, source_checkpoint, hf_layers, preserve_unloaded_prefixes
    )
    # Collect before ModelOpt's exporter temporarily replaces quantized modules.
    path.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=path, dtype=dtype,
                             save_modelopt_state=False, max_shard_size=max_shard_size,
                             extra_state_dict=extra_tensors or None)
    config, native_config, hashes = _validate_modelopt_export(path, hf_layers, preserved)
    config_path = path / "config.json"
    contract = {"quant_method": "residue_mxfp4", **FORMAT, "layers": runtime_layers}
    config["quantization_config"] = contract
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (path / "residue_mxfp4_config.json").write_text(json.dumps(contract, indent=2) + "\n")
    if processor is not None:
        processor.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    source_manifest = Path(__file__).resolve().parents[2] / "SOURCE_MANIFEST.json"
    manifest = {
        "status": "structure_validated", "runtime_validated": False,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "nvidia-modelopt", "transformers", "safetensors")},
        "native_quantization_config": native_config, "hf_layers": hf_layers,
        "source_checkpoint": source_record, "processor_saved": processor is not None,
        "native_packed_tensor_sha256": hashes, "preserved_parameter_sha256": preserved,
        "calibration": calibration, "tokenizer_saved": tokenizer is not None,
        "exporter_source_sha256": _file_hash(Path(__file__)),
        "source_manifest_sha256": _file_hash(source_manifest) if source_manifest.is_file() else None,
        "device": str(model_device),
        "cuda_capability": (list(torch.cuda.get_device_capability(model_device))
                            if model_device.type == "cuda" else None),
        "max_shard_size": max_shard_size,
        "files": {f.name: _file_hash(f) for f in sorted(path.iterdir()) if f.is_file()},
    }
    (path / "mxfp4_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path
