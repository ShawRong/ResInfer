# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'The RESIDUE_* environment contract, as one checkable object.'

from __future__ import annotations

import os
from typing import Mapping

METADATA_FILENAME = "residue_kernel_metadata.json"

# Values that read as "on" to a human but as OFF to `== "1"` parsing.
_TRUTHY_LOOKING = {"true", "yes", "on", "y", "enable", "enabled"}

# name -> (kind, allowed/None, one-line meaning)
#   kind: "one"    parsed with == "1"; anything else means OFF
#         "truthy" any non-empty value counts (plugin gate)
#         "enum"   must be one of `allowed` (case-sensitive)
#         "int"    must parse as int
#         "path"   filesystem path
#         "free"   uninterpreted string (csv lists, filenames)
KNOWN: dict[str, tuple] = {
    # -- activation / core -------------------------------------------------
    "RESIDUE_PATCH_VLLM": ("truthy", None, "plugin gate: install the patch"),
    "RESIDUE_VLLM_MODEL_DIR": ("path", None,
                               "export dir; metadata source, REQUIRED for "
                               "residue exports"),
    "RESIDUE_PATCH_MODE": ("enum", {"v2", "mext_r1"},
                           "patch flavor; mext_r1 also name-forces "
                           "metadata-less projections to runtime mext_r1"),
    "RESIDUE_VLLM_PATCH_VERSION": ("enum", {"v2", "mext_r1"},
                                   "legacy alias of RESIDUE_PATCH_MODE"),
    "RESIDUE_SWAP": ("one", None, "swapped-residue layout; must match export"),
    "RESIDUE_MEXT_PROJ_NAMES": ("free", None,
                                "csv name fragments the mext_r1 name-forcing "
                                "applies to"),
    "RESIDUE_MEXT_ATTN_PROJ_NAMES": ("free", None,
                                     "legacy alias of RESIDUE_MEXT_PROJ_NAMES"),
    # -- fold dispatch -----------------------------------------------------
    "RESIDUE_ENABLE_DECODE_MEXT": ("one", None,
                                   "fold layout for extended_k/plain layers"),
    "RESIDUE_DECODE_MEXT_MAX_M": ("int", None, "fold band M threshold"),
    "RESIDUE_FORCE_DECODE_MEXT": ("one", None,
                                  "fold at every M (PD decode-only servers)"),
    # EXPERIMENT: the attribution control for the per-layer ratio search.
    # Serves the SAME checkpoint with the prefill residue removed and decode
    # untouched. Declared here above all so it lands in the resolved line --
    # it is a server-side env, which is exactly the class of knob lighteval's
    # generation cache key cannot see, so two arms that differ only in this
    # would silently share a cache entry (.claude/rules/lighteval-cache.md).
    "RESIDUE_PREFILL_RESIDUE_OFF": ("one", None,
                                    "EXPERIMENT: no residue above the M "
                                    "threshold (decode keeps its residue)"),
    "RESIDUE_PREFILL_RESIDUE_OFF_MIN_M": ("int", None,
                                          "M threshold for the kill switch "
                                          "(default: the fold band)"),
    "RESIDUE_MEXT_PREFILL_PATH": ("enum",
                                  {"auto", "two_gemm", "k_loop"},
                                  'mext_r1 prefill route: auto selects FlashInfer-tuned two_gemm or k_loop; decode routing is independent.'),
    "RESIDUE_FOLD_MEXT": ("one", None, "enable A4/A6 tile8 families (sm10x)"),
    "RESIDUE_FOLD_SILENCE_DEFAULT": ("one", None,
                                     "silence the static-tactic-table warning"),
    "RESIDUE_DISABLE_SM10X_FOLD": ("one", None,
                                   "kill the sm10x fold (mext_r1 layers then "
                                   "refuse to serve, by design)"),
    "RESIDUE_MEXT_R1_LAYOUT": ("enum", {"concat", "row_pair", "tile8"},
                               "mext quant activation layout override"),
    "NVFP4_MEXT_R1_LAYOUT": ("enum", {"concat", "row_pair", "tile8"},
                             "legacy alias of RESIDUE_MEXT_R1_LAYOUT"),
    "NVFP4_MEXT_R1_ELTS_PER_THREAD": ("enum", {"auto", "8", "16"},
                                      "mext quant vector width"),
    # -- autotune / warmup ---------------------------------------------------
    "RESIDUE_SHARE_AUTOTUNE_CACHE": ("one", None,
                                     "share the FlashInfer autotune cache "
                                     "across models"),
    "RESIDUE_FOLD_TUNE_DUMP": ("free", None,
                               "write the fold tuner's decisions to this path"),
    # -- attention (experimental) -------------------------------------------
    "RESIDUE_ATTN": ("one", None, "register the ResAttn backend"),
    "RESIDUE_ATTN_CONTEXT_KERNEL": ("one", None,
                                    "0 disables the paged context kernel"),
    "RESIDUE_ATTN_PREFILL_EXACT": ("one", None,
                                   "force the gather_dequant oracle prefill"),
    # -- MoE -----------------------------------------------------------------
    "RESIDUE_MOE_MEXT": ("one", None, "MoE mext experiment gate"),
    "RESIDUE_MOE_OBSERVE": ("one", None, "MoE routing observer"),
    # -- debug ----------------------------------------------------------------
    "RESIDUE_DEBUG": ("one", None, "verbose debug prints"),
    "RESIDUE_DEBUG_MASK": ("one", None, "mask debug prints"),
    "RESIDUE_DEBUG_TP": ("one", None, "TP debug prints"),
    "RESIDUE_DEBUG_CREATE_WEIGHTS": ("one", None, "create_weights prints"),
    "RESIDUE_TENSOR_DUMP": ("free", None, "dump linear tensors to this dir"),
    "RESIDUE_STACK_DUMP_SIGNAL": ("one", None,
                                  "arm SIGUSR1 all-thread stack dumps"),
    "RESIDUE_ENV_CHECK": ("enum", {"0", "1"},
                          "0 downgrades contract refusals to warnings"),
}

# Set in tests / internal tooling; never a typo worth warning about.
_IGNORED_PREFIXES = ("RESIDUE_TEST_",)


def check(env: Mapping[str, str], model_dir: str | None = None) -> dict:
    """Pure rule evaluation. Returns {resolved, warnings, fatals}.

    `model_dir` defaults to env's RESIDUE_VLLM_MODEL_DIR. Filesystem access is
    limited to two existence probes on it.
    """
    warnings: list[str] = []
    fatals: list[str] = []

    def val(name: str) -> str | None:
        return env.get(name)

    # R1: unknown RESIDUE_* -- typos silently do nothing.
    for name in sorted(env):
        if not name.startswith("RESIDUE_"):
            continue
        if name in KNOWN or name.startswith(_IGNORED_PREFIXES):
            continue
        warnings.append(
            f"unknown variable {name} -- nothing reads it (typo?)")

    # R2/R7/R8/R9: per-variable value checks.
    for name, (kind, allowed, _desc) in KNOWN.items():
        raw = val(name)
        if raw is None:
            continue
        if kind == "one" and raw != "1":
            if raw.strip().lower() in _TRUTHY_LOOKING:
                warnings.append(
                    f"{name}={raw!r} parses as OFF -- this switch only "
                    f"understands '1'")
        elif kind == "enum" and raw not in (allowed or set()):
            fatals.append(
                f"{name}={raw!r} is not one of {sorted(allowed)} -- an "
                f"unrecognized value would silently fall back")
        elif kind == "int":
            try:
                int(raw)
            except ValueError:
                fatals.append(
                    f"{name}={raw!r} is not an integer -- it is parsed with "
                    f"int() deep in the dispatch and would crash there")

    model_dir = model_dir or val("RESIDUE_VLLM_MODEL_DIR")
    has_metadata = False
    if model_dir:
        if not os.path.isdir(model_dir):
            # Incident: a wrong path behaves exactly like an unset one.
            fatals.append(
                f"RESIDUE_VLLM_MODEL_DIR={model_dir!r} does not exist")
        else:
            has_metadata = os.path.isfile(
                os.path.join(model_dir, METADATA_FILENAME))
    else:
        
        
        warnings.append(
            "RESIDUE_VLLM_MODEL_DIR is unset -- if the checkpoint is a "
            "residue export, create_weights will size merged params at stock "
            "K and weight loading dies on `assert param_data.shape == "
            "loaded_weight.shape`. Set it to the export directory.")

    mode = val("RESIDUE_PATCH_MODE") or val("RESIDUE_VLLM_PATCH_VERSION")

    if has_metadata and mode == "mext_r1":
        
        
        
        warnings.append(
            "RESIDUE_PATCH_MODE=mext_r1 while the export carries "
            f"{METADATA_FILENAME}: any projection WITHOUT a metadata entry is "
            "silently name-forced to runtime mext_r1. For exports you almost "
            "always want the mode unset (v2).")

    if has_metadata and val("RESIDUE_ENABLE_DECODE_MEXT") != "1":
        
        
        
        
        
        
        warnings.append(
            "residue export without RESIDUE_ENABLE_DECODE_MEXT=1: on vLLM "
            "0.26 this crashed weight loading with KeyError "
            "'weight_scale_base' (implementation constraints, Still "
            "open). If loading fails there, set RESIDUE_ENABLE_DECODE_MEXT=1.")


    resolved = (
        f"mode={mode or 'v2(default)'} "
        f"model_dir={'set' if model_dir else 'UNSET'}"
        f"{'+metadata' if has_metadata else ''} "
        f"decode_mext={'on' if val('RESIDUE_ENABLE_DECODE_MEXT') == '1' else 'off'} "
        f"max_m={val('RESIDUE_DECODE_MEXT_MAX_M') or 'default'} "
        f"prefill_path={val('RESIDUE_MEXT_PREFILL_PATH') or 'auto'} "
        f"prefill_residue="
        f"{'OFF' if val('RESIDUE_PREFILL_RESIDUE_OFF') == '1' else 'on'} "
        f"share_autotune={'on' if val('RESIDUE_SHARE_AUTOTUNE_CACHE') == '1' else 'off'} "
        f"swap={'on' if val('RESIDUE_SWAP') == '1' else 'off'} "
        f"attn={'on' if val('RESIDUE_ATTN') == '1' else 'off'}"
    )
    return {"resolved": resolved, "warnings": warnings, "fatals": fatals}


def enforce(model_dir: str | None = None) -> None:
    """Check os.environ, print the resolved line, refuse fatal combinations.

    Called from patch_vllm_modelopt in EVERY vLLM process (the plugin runs in
    the spawned EngineCore too). RESIDUE_ENV_CHECK=0 downgrades refusals to
    warnings; nothing disables the resolved-config line.
    """
    report = check(os.environ, model_dir=model_dir)
    print(f"[residue:env] {report['resolved']}", flush=True)
    for w in report["warnings"]:
        print(f"[residue:env] WARNING: {w}", flush=True)
    if not report["fatals"]:
        return
    if os.environ.get("RESIDUE_ENV_CHECK") == "0":
        for f in report["fatals"]:
            print(f"[residue:env] (downgraded) FATAL: {f}", flush=True)
        return
    lines = "\n  - ".join(report["fatals"])
    raise RuntimeError(
        f"[residue:env] refusing to start with a known-fatal configuration:"
        f"\n  - {lines}\n"
        "Fix the variables above, or set RESIDUE_ENV_CHECK=0 to run anyway "
        "(the failure mode is documented next to each rule in "
        "quant/vllm_integration/env_contract.py)."
    )
