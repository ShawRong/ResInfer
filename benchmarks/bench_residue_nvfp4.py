#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Reproducible kernel benchmark for Residue NVFP4 linear layers.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import torch

from _common import fi_autotuned, timed_detail

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "csrc" / "nvfp4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ----------------------------------------------------------------- shapes
# (N, K) for x[M,K] @ W[K,N] -> y[M,N], i.e. the vLLM *fused* projections.
# Read off the public configs; fused sizes are what actually gets launched.
SHAPES: dict[str, dict[str, tuple[int, int]]] = {
    "qwen3_4b": {                       # hidden 2560, inter 9728, 36 layers
        "qkv": (6144, 2560), "o": (2560, 4096),
        "gate_up": (19456, 2560), "down": (2560, 9728),
    },
    "qwen3_8b": {                       # hidden 4096, inter 12288, 36 layers
        "qkv": (6144, 4096), "o": (4096, 4096),
        "gate_up": (24576, 4096), "down": (4096, 12288),
    },
    "qwen3_32b": {                      # hidden 5120, inter 25600, 64 layers
        "qkv": (10240, 5120), "o": (5120, 8192),
        "gate_up": (51200, 5120), "down": (5120, 25600),
    },
}




SUPPORTED_RATIOS = (0.125, 0.25, 0.5)

# Dense where the decode band is, thinning out above it. The default ceiling
# is per-arch: the fold band is 128 on sm10x and 64 elsewhere, and the sweep
# needs enough headroom above it to show what happens past the band without
# spending the whole run there.
M_GRID = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128,
          160, 192, 256, 320, 384, 448, 512)
DEFAULT_M_MAX = {"sm100": 512, "sm103": 512, "sm120": 128}


# ------------------------------------------------------------- provenance
def _md5(path: pathlib.Path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:                                           # noqa: BLE001
        return None


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:                                           # noqa: BLE001
        try:
            from importlib.metadata import version
            return version(mod)
        except Exception:                                       # noqa: BLE001
            return None


def _nvsmi(index: int, fields: str) -> list[str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-{fields}",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    except Exception:                                           # noqa: BLE001
        return []


def physical_index(logical: int) -> int:
    """Map a torch device index to the index `nvidia-smi` understands.

    Under CUDA_VISIBLE_DEVICES=6, torch's device 0 is physical GPU 6 while
    `nvidia-smi -i 0` is still physical GPU 0. Querying the wrong one reported
    a neighbouring user's training job as this benchmark's neighbours -- and,
    worse, would report an empty GPU as clean while the real one was busy.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        return logical
    ids = [v.strip() for v in vis.split(",") if v.strip()]
    try:
        return int(ids[logical])
    except (IndexError, ValueError):
        return logical


def gpu_state(index: int) -> dict:
    """Clocks, temperature and neighbour count.

    Neighbours matter: a benchmark sharing a GPU with any other compute
    process is not measuring this kernel. Clocks are recorded rather than
    locked because locking needs privileges a reproducer may not have; a
    thermal drift then shows up as a column instead of an unexplained row.
    """
    index = physical_index(index)
    out: dict = {"index": index}
    rows = _nvsmi(index, "gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,"
                         "power.draw,memory.used,memory.total")
    if rows:
        keys = ("name", "uuid", "clock_sm_mhz", "clock_mem_mhz", "temp_c",
                "power_w", "mem_used_mib", "mem_total_mib")
        out.update(dict(zip(keys, [v.strip() for v in rows[0].split(",")])))
    apps = _nvsmi(index, "compute-apps=pid,used_memory")
    out["neighbors"] = max(0, len(apps) - 1)   # this process is one of them
    return out


TRACKED_ENV = (
    "RESIDUE_PATCH_VLLM", "RESIDUE_PATCH_MODE", "RESIDUE_ENABLE_DECODE_MEXT",
    "RESIDUE_FORCE_DECODE_MEXT", "RESIDUE_DECODE_MEXT_MAX_M",
    "RESIDUE_MEXT_PREFILL_PATH", "RESIDUE_MEXT_R1_LAYOUT",
    "RESIDUE_PREFILL_RESIDUE_OFF", "RESIDUE_TENSOR_DUMP", "RESIDUE_SWAP",
    "VLLM_DISABLE_COMPILE_CACHE", "VLLM_BATCH_INVARIANT", "VLLM_PLUGINS",
    "FLASHINFER_CUBIN_DIR", "CUDA_VISIBLE_DEVICES",
)


def provenance(gpu: int, run_id: str) -> dict:
    so = next(iter(sorted(
        (REPO_ROOT / "csrc" / "nvfp4" / "nvfp4").glob("_C*.so"))), None)
    major, minor = torch.cuda.get_device_capability(gpu)
    return {
        "kind": "provenance",
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_subject": _git("log", "-1", "--format=%s"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "ext_so": str(so) if so else None,
        "ext_so_mtime": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(so.stat().st_mtime))
                         if so else None),
        "ext_so_md5": _md5(so) if so else None,
        "sm": f"sm{major}{minor}",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _version("torch"),
        "vllm": _version("vllm"),
        "flashinfer": _version("flashinfer"),
        "cuda": torch.version.cuda,
        "gpu": gpu_state(gpu),
        "env": {k: os.environ.get(k, "<unset>") for k in TRACKED_ENV},
    }


# ------------------------------------------------------------------ setup
def global_scale(t: torch.Tensor) -> torch.Tensor:
    """vLLM/CUTLASS FP4 convention: 448*6/amax, passed into the quantiser."""
    return (448.0 * 6.0 / t.abs().max().float()).reshape(())


def make_weight(n: int, k: int, device: torch.device, dtype: torch.dtype):
    """Quantise a random weight the way vLLM does, and return both forms.

    Both the packed FP4 weight and the bf16 tensor it came from: the latter is
    the reference the correctness check compares against, so it must be the
    exact tensor that was quantised, not a fresh draw.
    """
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale)

    w_bf16 = torch.randn(n, k, device=device, dtype=dtype) / (k ** 0.5)
    w_gs = global_scale(w_bf16)
    w_fp4, w_sf = scaled_fp4_quant(w_bf16, w_gs)
    return w_bf16, w_fp4, swizzle_blockscale(w_sf), w_gs


class FakeLayer(torch.nn.Module):
    """The attribute surface `NvFp4LinearKernel.apply_weights` reads.

    Deliberately minimal and deliberately NOT a reimplementation: vLLM's own
    kernel object does the work, this only supplies what it looks up.
    """

    def __init__(self, weight, weight_scale, alpha, input_gs, out_size,
                 weight_gs=None):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.alpha = alpha
        self.input_global_scale_inv = input_gs
        self.output_size_per_partition = out_size
        self.weights_padding_cols = 0
        # Unused by the fast kernels, which fold both global scales into
        # `alpha`. Kept because vLLM's emulation kernel reads it, so this
        # layer can be handed to that path unchanged if a reference is ever
        # wanted.
        self.weight_global_scale = weight_gs


def salient_indices(k: int, ratio: float, device: torch.device) -> torch.Tensor:
    """The per-8-channel salient pattern the export format uses.

    Selection is structurally confined to each 8-channel block
    (`residue_per_8`), so a benchmark that picked globally would be measuring
    a layout the exporter cannot produce. Which channels within the block are
    picked does not affect cost, so a deterministic prefix is used.
    """
    if k % 8:
        raise ValueError(f"k must be a multiple of 8, got {k}")
    per_block = int(round(ratio * 8))
    if per_block == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    base = torch.arange(0, k, 8, dtype=torch.long, device=device)
    off = torch.arange(per_block, dtype=torch.long, device=device)
    return (base[:, None] + off[None, :]).reshape(-1).contiguous()


# ----------------------------------------------------------------- timing
class TacticWatch:
    'Record whether the fold ran on the STATIC tactic table.'

    MARK = "STATIC tactic table"

    def __init__(self) -> None:
        self.hits: list[str] = []

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        for w in self._log:
            text = str(w.message)
            if self.MARK in text:
                self.hits.append(" ".join(text.split())[:400])
        self._cm.__exit__(*exc)
        return False



def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()


def rel_mae(ref: torch.Tensor, out: torch.Tensor) -> float:
    d = (ref.float() - out.float()).abs().mean().item()
    denom = ref.float().abs().mean().item()
    return d / denom if denom else float("nan")


# ------------------------------------------------------------------- main
def run(args) -> None:
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale)
    import quant.vllm_integration.residue_linear_op  # noqa: F401  registers op
    from nvfp4 import fold as nvfp4_fold

    stock_kernel = init_nvfp4_linear_kernel()
    stock_name = type(stock_kernel).__name__

    major, minor = torch.cuda.get_device_capability(device)
    sm = f"sm{major}{minor}"
    m_max = args.m_max or DEFAULT_M_MAX.get(sm, 128)
    m_grid = [m for m in M_GRID if m <= m_max]

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec: dict) -> None:
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    fold_seen = [False]      # sticky; see the fold_fired check below
    prov = provenance(args.gpu, run_id)
    prov["stock_kernel_class"] = stock_name
    prov["m_grid"] = m_grid
    prov["ratios"] = list(args.ratios)
    emit(prov)
    print(f"[bench] {sm}  stock={stock_name}  M<= {m_max}  "
          f"commit={prov['git_commit'][:12] if prov['git_commit'] else '?'}")
    if prov["gpu"].get("neighbors"):
        print(f"[bench] WARNING: {prov['gpu']['neighbors']} other compute "
              f"process(es) on this GPU -- timings are not trustworthy.")

    for model in args.models:
        for proj, (n, k) in SHAPES[model].items():
            w_bf16, w_fp4, w_sf, w_gs = make_weight(n, k, device, dtype)
            base_layer = FakeLayer(w_fp4, w_sf, None, None, n, weight_gs=w_gs)

            # kext weights: K grows to K*(1+r). Built once per ratio and
            # reused across M so the per-M timings differ only in M.
            kext = {}
            for r in args.ratios:
                idx = salient_indices(k, r, device)
                w_ext_bf16 = torch.cat(
                    [w_bf16, w_bf16.index_select(-1, idx)], dim=-1)
                gs = global_scale(w_ext_bf16)
                fp4, sf = scaled_fp4_quant(w_ext_bf16, gs)
                kext[r] = dict(w=fp4, sf=swizzle_blockscale(sf), gs=gs,
                               idx=idx, k_ext=w_ext_bf16.shape[-1])

            for m in m_grid:
                x = torch.randn(m, k, device=device, dtype=dtype)
                x_gs = global_scale(x)
                ref = (x.float() @ w_bf16.float().T).to(dtype)

                # ---- stock: vLLM's own selector, vLLM's own kernel object
                base_layer.alpha = (1.0 / (x_gs * w_gs)).reshape(())
                base_layer.input_global_scale_inv = x_gs
                stock_fn = lambda *_: stock_kernel.apply_weights(  # noqa: E731
                    base_layer, x)
                # Give the baseline its tuned tactic before timing it.
                # FlashInfer's whole advantage is a per-shape search, so a cold
                # cache does not measure FlashInfer -- it measures FlashInfer
                # handicapped, and every residue number would inherit the
                # flattery. This is the mirror of not letting our own fold run
                # on the static fallback table: both sides get warmed, and both
                # facts are recorded.
                if args.autotune:
                    fi_autotuned(stock_fn, (x, base_layer.weight), m_bucket=m)
                stock = timed_detail(stock_fn, (x, base_layer.weight),
                                     repeats=args.repeats)
                y_stock = stock_kernel.apply_weights(base_layer, x)
                emit({"kind": "measure", "run_id": run_id, "family": "stock",
                      "rep": args.rep, "model": model, "proj": proj, "n": n, "k": k, "m": m,
                      "ratio": 0.0, "stock_kernel_class": stock_name,
                      # Context, not a gate: ~0.94 for every arm including
                      # this one, because it is NVFP4's quantisation error
                      # rather than anything about the kernel.
                      "cos_vs_bf16": cosine(ref, y_stock),
                      "rel_mae_vs_bf16": rel_mae(ref, y_stock),
 **stock})

                placeholder = torch.zeros(1, dtype=torch.uint8, device=device)

                # ---- mext: ratio 1.0, weight stored once, rows doubled
                if args.family in ("mext", "both"):
                    f = lambda *_: torch.ops.residue.nvfp4_linear(   # noqa: E731
                        x, w_fp4, x_gs, w_sf, w_sf, placeholder,
                        (1.0 / (x_gs * w_gs)).reshape(()),
                        k, k, 0, n, True, False, True)
                    # Compile OUTSIDE the counted window. sm120 JITs the fold
                    # per (n_w, k, m_tok, tile, a_ld) at ~1.3 s a shape; taking
                    # the snapshot before warmup counted that compile as if it
                    # had happened during timing and flagged every first row.
                    # The residue stack prints when the fold really runs.
                    # Capturing it is the manipulation check: a residue arm
                    # that quietly served plain NVFP4 would otherwise report a
                    # flattering number and look entirely normal.
                    import contextlib
                    import io
                    buf = io.StringIO()
                    with TacticWatch() as tw, contextlib.redirect_stdout(buf):
                        f()
                        torch.cuda.synchronize()
                    # The stack announces the fold ONCE per process, not per
                    # call, so this is sticky: False on later rows would be a
                    # false negative, and a manipulation check that cries wolf
                    # is one nobody reads. What it certifies is the run, not
                    # the row -- if it is False for a whole run, no residue arm
                    # in that file folded and the file is not what it claims.
                    if "fold FIRED" in buf.getvalue():
                        fold_seen[0] = True
                    before = nvfp4_fold.compile_stats()
                    t = timed_detail(f, (x, w_fp4), repeats=args.repeats)
                    y = f()
                    after = nvfp4_fold.compile_stats()
                    for hit in tw.hits:
                        emit({"kind": "tactic_fallback", "run_id": run_id,
                              "family": "mext", "rep": args.rep, "model": model, "proj": proj,
                              "n": n, "k": k, "m": m, "warning": hit})
                    emit({"kind": "measure", "run_id": run_id, "family": "mext",
                          "rep": args.rep, "model": model, "proj": proj, "n": n, "k": k, "m": m,
                          "ratio": 1.0, "stock_kernel_class": stock_name,
                          "vs_stock": t["us"] / stock["us"],
                          "vs_stock_min": t["p50_us"] / stock["p50_us"],
                          "cos_vs_bf16": cosine(ref, y),
                          "rel_mae_vs_bf16": rel_mae(ref, y),

                          # Non-zero means the timing window paid for a JIT and
                          # the row is not a measurement of the kernel.
                          "jit_during_timing": _compiled_delta(before, after),
                          "static_tactic_table": bool(tw.hits),
                          "fold_fired_in_run": fold_seen[0],
                          **t})

                # ---- kext: ratio r < 1, K grows to K*(1+r)
                if args.family in ("kext", "both"):
                    for r in args.ratios:
                        e = kext[r]
                        alpha = (1.0 / (x_gs * e["gs"])).reshape(())
                        f = lambda *_, e=e, alpha=alpha: torch.ops.residue.nvfp4_linear(  # noqa: E731
                            x, e["w"], x_gs, e["sf"], e["sf"], placeholder,
                            alpha, k, int(e["idx"].numel()), 0, n,
                            False, False, False)
                        t = timed_detail(f, (x, e["w"]), repeats=args.repeats)
                        y = f()
                        emit({"kind": "measure", "run_id": run_id,
                              "family": "kext", "rep": args.rep, "model": model, "proj": proj,
                              "n": n, "k": k, "m": m, "ratio": r,
                              "k_ext": e["k_ext"],
                              "stock_kernel_class": stock_name,
                              # absolute: what a deployment actually pays
                              "vs_stock": t["us"] / stock["us"],
                              # normalised: (1+r) is the arithmetic floor, so
                              # this column is ~1.0 iff we pay nothing beyond
                              # the arithmetic the extension implies
                              "vs_ideal": t["us"] / ((1.0 + r) * stock["us"]),
                              "cos_vs_bf16": cosine(ref, y),
                              "rel_mae_vs_bf16": rel_mae(ref, y),
                              **t})

                del x
            del w_bf16, w_fp4, w_sf, kext
            torch.cuda.empty_cache()
            print(f"[bench] done {model}/{proj}  N={n} K={k}")

    emit({"kind": "run_summary", "run_id": run_id, "rep": args.rep,
          "baseline_autotuned_per_m": bool(args.autotune),
          "fold_fired_in_run": fold_seen[0],
          "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "gpu_end": gpu_state(args.gpu)})
    print(f"[bench] wrote {out}")


def _compiled_delta(before: dict, after: dict) -> int:
    """Fold kernels JIT-compiled between two `compile_stats()` snapshots.

    Non-zero means the timing window paid for a compile (~1.3 s on sm120,
    which JITs per shape and M). Such a row is not a measurement of the
    kernel; `analyze` drops it.
    """
    total = 0
    for arch in ("sm10x", "sm120"):
        b, a = before.get(arch), after.get(arch)
        if a:
            total += a["compiled"] - (b["compiled"] if b else 0)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--models", default="qwen3_4b,qwen3_8b,qwen3_32b",
                   help=f"comma-separated: {', '.join(SHAPES)}")
    p.add_argument("--family", default="both", choices=["mext", "kext", "both"])
    p.add_argument("--ratios", default="0.125,0.25,0.5")
    p.add_argument("--m-max", type=int, default=None,
                   help="default is per-arch: 512 on sm100/sm103, 128 on sm120")
    p.add_argument("--repeats", type=int, default=5,
                   help="cudagraph medians per point; the MIN is "
                        "the reported number, the spread is the "
                        "within-process noise floor")
    p.add_argument("--rep", type=int, default=0,
                   help="index of this WHOLE-PROCESS repetition. "
                        "Within-process medians share one autotune "
                        "decision and one allocation layout, so "
                        "run-to-run needs the process re-run; the "
                        "driver loops this and rows carry the index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-autotune", dest="autotune", action="store_false",
                   help="skip FlashInfer tuning of the BASELINE; recorded "
                        "in provenance because it changes what stock means")
    p.set_defaults(autotune=True)
    p.add_argument("--out", default="residue_kernel_bench.jsonl")
    a = p.parse_args()
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]
    a.ratios = [float(r) for r in a.ratios.split(",") if r.strip()]
    for r in a.ratios:
        if not any(abs(r - s) < 1e-9 for s in SUPPORTED_RATIOS):
            raise SystemExit(f"unsupported ratio {r}; expected one of "
                             f"{SUPPORTED_RATIOS} (1.0 is the mext family)")
    for m in a.models:
        if m not in SHAPES:
            raise SystemExit(f"unknown model {m}; expected one of {list(SHAPES)}")
    run(a)


if __name__ == "__main__":
    main()
