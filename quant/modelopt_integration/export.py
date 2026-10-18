"""Common ModelOpt export: native quantization config plus residue metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from quant.kernel_metadata import padded_k, residue_spec, runtime_layer_specs

def export_residue_checkpoint(
    model, export_dir, dtype=None, residue_ratio=0.125, *,
    block_size=8, calibration=None, source_checkpoint=None,
    preserve_unloaded_prefixes=None, tokenizer=None, processor=None,
    save_kernel_metadata=True, save_channel_masks=True, **export_kwargs,
):
    """Use ModelOpt's exporter for both NVFP4 and MXFP4, without changing format.

    Collect residue metadata before ModelOpt temporarily packs/replaces modules.
    MXFP4 validation checks native output; it never rewrites weights or config.
    """
    from modelopt.torch.export import export_hf_checkpoint
    from .kernel_metadata import _collect_export_metadata, _write_kernel_metadata

    path = Path(export_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite export: {path}")
    mxfp4 = has_mxfp4_weights(model)
    metadata, mask_dims = _collect_export_metadata(
        model, residue_ratio, block_size, save_channel_masks, calibration,
    )
    hf_layers, preserved, source_record = {}, {}, None
    if mxfp4:
        _require_modelopt_export_api(export_hf_checkpoint)
        model_type = getattr(model.config, "model_type", None)
        if model_type == "qwen3_5" and source_checkpoint is None:
            raise ValueError("Qwen3.5 requires the source checkpoint to preserve vision/MTP")
        hf_layers = _collect_mxfp4_layers(model, metadata["layers"])
        # Validate fused activation contracts before writing anything.
        runtime_layer_specs(hf_layers, model_type)
    nvfp4_shapes = {}
    if source_checkpoint is not None or mxfp4:
        if preserve_unloaded_prefixes is None:
            preserve_unloaded_prefixes = (("mtp.",) if
                getattr(model.config, "model_type", None) == "qwen3_5" else ())
        quantized_layers = hf_layers or {
            name: {} for name, module in model.named_modules()
            if getattr(getattr(module, "weight_quantizer", None), "is_enabled", False)
        }
        if not mxfp4:
            nvfp4_shapes = {name: list(module.weight.shape)
                           for name, module in model.named_modules()
                           if name in quantized_layers}
        extra, preserved, source_record = _collect_preserved_state(
            model, source_checkpoint, quantized_layers, preserve_unloaded_prefixes)
        if extra:
            supplied = export_kwargs.get("extra_state_dict") or {}
            if set(extra) & set(supplied):
                raise ValueError("duplicate preserved extra_state_dict keys")
            export_kwargs["extra_state_dict"] = {**supplied, **extra}

    path.mkdir(parents=True, exist_ok=True)
    # Required residue state must never depend on an optional debug flag.
    if save_kernel_metadata or metadata["layers"] or metadata["extended_linear_dims"]:
        _write_kernel_metadata(metadata, mask_dims, path)
    export_kwargs.setdefault("save_modelopt_state", False if mxfp4 else True)
    if mxfp4:
        export_kwargs["save_modelopt_state"] = False
    with torch.inference_mode(), _omit_private_export_buffers(model):
        export_hf_checkpoint(model, export_dir=path, dtype=dtype, **export_kwargs)
    _normalize_quant_algo_for_vllm(path)
    manifest = None
    if mxfp4 or source_checkpoint is not None:
        if mxfp4:
            _, native, hashes = _validate_modelopt_export(path, hf_layers, preserved)
        else:
            native, hashes = _validate_nvfp4_export(path, nvfp4_shapes, preserved)
        manifest = {
            "status": "structure_validated", "runtime_validated": False,
            "native_quantization_config": native,
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "source_checkpoint": source_record,
            "max_shard_size": export_kwargs.get("max_shard_size", "10GB"),
            "tokenizer_saved": tokenizer is not None,
            "processor_saved": processor is not None,
            "exporter_source_sha256": _file_hash(Path(__file__)),
            "native_packed_tensor_sha256": hashes,
            "preserved_parameter_sha256": preserved,
            "calibration": calibration,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("torch", "nvidia-modelopt", "transformers", "safetensors")},
        }
    if processor is not None:
        processor.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    if manifest is not None:
        manifest["metadata_file_sha256"] = {
            file.name: _file_hash(file) for file in sorted(path.iterdir())
            if file.is_file() and file.suffix != ".safetensors"
        }
        manifest["safetensor_shards"] = {
            file.name: file.stat().st_size for file in path.glob("*.safetensors")
        }
        source_manifest = Path(__file__).resolve().parents[2] / "SOURCE_MANIFEST.json"
        manifest["source_manifest_sha256"] = (_file_hash(source_manifest)
                                              if source_manifest.is_file() else None)
        device = next(model.parameters()).device
        manifest["device"] = str(device)
        manifest["cuda_capability"] = (list(torch.cuda.get_device_capability(device))
                                       if device.type == "cuda" else None)
        (path / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _validate_nvfp4_export(path, shapes, preserved):
    """Check the packed inventory and preserved source scopes for NVFP4."""
    from safetensors import safe_open

    config = json.loads((path / "config.json").read_text())
    native = config.get("quantization_config", {})
    if (native.get("quant_method") != "modelopt" or
            str(native.get("quant_algo", "")).upper() not in ("NVFP4", "FP4")):
        raise ValueError(f"unexpected ModelOpt NVFP4 config: {native!r}")
    shards = sorted(path.glob("*.safetensors"))
    actual, hashes = {}, {}
    expected = set(preserved) | {f"{name}.{suffix}" for name in shapes
                                for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name in actual or "quantizer." in name or "_salient_indices" in name:
                    raise ValueError(f"duplicate or private exported tensor: {name}")
                actual[name] = shard.name
                tensor = handle.get_tensor(name)
                if tensor.is_floating_point() and not tensor.float().isfinite().all().item():
                    raise ValueError(f"nonfinite exported tensor: {name}")
                if name in preserved and _hash(tensor) != preserved[name]:
                    raise ValueError(f"unquantized parameter changed during export: {name}")
                prefix, _, leaf = name.rpartition(".")
                if prefix in shapes:
                    rows, cols = shapes[prefix]
                    if leaf == "weight" and (tensor.dtype != torch.uint8 or
                                              list(tensor.shape) != [rows, cols // 2]):
                        raise ValueError(f"invalid packed NVFP4 weight: {name} {tensor.shape}")
                    if leaf == "weight_scale" and list(tensor.shape) != [rows, cols // 16]:
                        raise ValueError(f"invalid NVFP4 block scales: {name} {tensor.shape}")
                    hashes[name] = _hash(tensor)
    if not shards or expected - set(actual):
        raise ValueError(f"missing NVFP4 tensors: {sorted(expected - set(actual))[:20]}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        if json.loads(index.read_text()).get("weight_map") != actual:
            raise ValueError("ModelOpt shard index does not match exported tensors")
    elif len(shards) != 1:
        raise ValueError("multi-shard ModelOpt export is missing its safetensors index")
    return native, hashes


def _normalize_quant_algo_for_vllm(export_path: Path) -> None:
    """Rewrite exported quant_algo to "NVFP4" for vLLM compatibility.

    This is a lightweight metadata patch (does NOT touch any weights).
    It preserves the original quant_algo under an auxiliary key for traceability.
    """

    def _patch_json(path: Path, *, key_path: list[str], aux_key: str) -> None:
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"[residue] Warning: Failed to read {path}: {e}")
            return

        cur = data
        for k in key_path[:-1]:
            if not isinstance(cur, dict) or k not in cur:
                return
            cur = cur[k]
        leaf = key_path[-1]
        if not isinstance(cur, dict) or leaf not in cur:
            return

        quant_algo = cur.get(leaf)
        if not isinstance(quant_algo, str):
            return

        if quant_algo.upper() != "NVFP4_AWQ":
            return

        cur[aux_key] = quant_algo
        cur[leaf] = "NVFP4"
        try:
            path.write_text(json.dumps(data, indent=4) + "\n")
        except Exception as e:
            print(f"[residue] Warning: Failed to write {path}: {e}")

    # config.json: quantization_config.quant_algo
    _patch_json(
        export_path / "config.json",
        key_path=["quantization_config", "quant_algo"],
        aux_key="_original_quant_algo",
    )
    # hf_quant_config.json: quantization.quant_algo
    _patch_json(
        export_path / "hf_quant_config.json",
        key_path=["quantization", "quant_algo"],
        aux_key="_original_quant_algo",
    )



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


def _collect_mxfp4_layers(model, metadata):
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
        k = int(getattr(module, "_original_in_features", module.weight.shape[1]))
        spec = residue_spec(metadata.get(name), k, module.weight.shape[0])
        expected = padded_k(k, len(spec.get("indices", [])))
        if module.weight.shape[1] != expected:
            raise ValueError(f"materialize and pad selected weights before export: {name}; "
                             f"expected K={expected}, got {module.weight.shape[1]}")
        if not torch.isfinite(module.weight).all().item():
            raise ValueError(f"nonfinite source weight: {name}")
        layers[name] = spec
    if not layers:
        raise ValueError("no enabled MXFP4 linear layers found")
    return layers


@contextmanager
def _omit_private_export_buffers(model):
    """Keep residue metadata live while excluding it from exported state_dicts."""
    added = []
    for module in model.modules():
        if "_salient_indices" not in getattr(module, "_buffers", {}):
            continue
        non_persistent = module._non_persistent_buffers_set
        if "_salient_indices" not in non_persistent:
            non_persistent.add("_salient_indices")
            added.append(module)
    try:
        yield
    finally:
        for module in added:
            module._non_persistent_buffers_set.discard("_salient_indices")


def _validate_modelopt_export(path, hf_layers, preserved):
    """Validate official ModelOpt output without rewriting its shards."""
    from safetensors import safe_open

    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    native_config = config.get("quantization_config")
    if (not isinstance(native_config, dict)
            or native_config.get("quant_method") != "modelopt"
            or str(native_config.get("quant_algo", "")).lower() != "mxfp4"):
        raise ValueError(f"unexpected ModelOpt MXFP4 config: {native_config!r}")

    seen, hashes, weight_map = set(), {}, {}
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
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"duplicate exported tensor: {key}")
                seen.add(key)
                if "quantizer." in key:
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
                            "unexpected native ModelOpt MXFP4 shape/dtype: "
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
                weight_map[key] = shard.name

    if native_index is not None:
        if native_weight_map != weight_map:
            raise ValueError("ModelOpt shard index does not match exported tensors")
    elif len(shards) != 1:
        raise ValueError("multi-shard ModelOpt export is missing its safetensors index")
    missing = (expected_quant | set(preserved)) - set(weight_map)
    if missing:
        raise ValueError(f"missing exported tensors: {sorted(missing)}")
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



def is_residue_quantized(model: torch.nn.Module) -> bool:
    """Check if model has residue quantization applied.

    Detects residue quantization by looking for the _original_in_features
    attribute that is set during residue weight extension.

    Args:
        model: Model to check

    Returns:
        True if model has residue quantization applied
    """
    for module in model.modules():
        if hasattr(module, "_original_in_features"):
            return True
    return False
