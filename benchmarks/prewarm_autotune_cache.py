#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Populate the FlashInfer autotune cache offline, before anything is served.'

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

FP4_MAX = 448.0 * 6.0







_FUSE_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),   # -> qkv_proj
    ("gate_proj", "up_proj"),          # -> gate_up_proj
)


def shapes_from_export(export_dir: str) -> set[tuple[int, int]]:
    """Distinct GEMM (n_out, k_logical) shapes a server will actually run.

    Fuses qkv and gate_up the way vLLM does: same K required (a residue
    export extends every member of a fused group identically, or the server
    could not fuse them either); n_out is the sum over members.
    """
    from safetensors import safe_open

    raw: dict[str, tuple[int, int]] = {}
    for f in sorted(glob.glob(os.path.join(export_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as fh:
            for key in fh.keys():
                if not (key.endswith(".weight") and "layers" in key):
                    continue
                if "norm" in key:
                    continue
                shape = fh.get_slice(key).get_shape()
                if len(shape) == 2:
                    # Stored FP4-packed: the logical K is twice the stored one.
                    raw[key[: -len(".weight")]] = (
                        int(shape[0]), int(shape[1]) * 2)

    by_prefix: dict[str, dict[str, tuple[int, int]]] = {}
    for name, shp in raw.items():
        prefix, _, proj = name.rpartition(".")
        by_prefix.setdefault(prefix, {})[proj] = shp

    out: set[tuple[int, int]] = set()
    for prefix, projs in by_prefix.items():
        fused: set[str] = set()
        for members in _FUSE_GROUPS:
            if not all(m in projs for m in members):
                continue
            ks = {projs[m][1] for m in members}
            if len(ks) != 1:
                print(f"WARNING: {prefix} {members} have unequal K {ks}; "
                      "a server could not fuse these either -- keeping them "
                      "unfused", file=sys.stderr)
                continue
            out.add((sum(projs[m][0] for m in members), ks.pop()))
            fused |= set(members)
        for proj, shp in projs.items():
            if proj not in fused:
                out.add(shp)
    return out


def shared_cache_path() -> pathlib.Path:
    """The path RESIDUE_SHARE_AUTOTUNE_CACHE=1 makes the server read.

    Mirrors vLLM's layout (<root>/<fi_version>/<arch>/) with "shared" where the
    per-model hash would be. Kept in sync with
    quant/vllm_integration/patch.py:_install_shared_autotune_cache.
    """
    from flashinfer.jit import env as fi_env

    ws = fi_env.FLASHINFER_WORKSPACE_DIR
    root = pathlib.Path(
        os.environ.get("VLLM_CACHE_ROOT",
                       os.path.expanduser("~/.cache/vllm"))
    ) / "flashinfer_autotune_cache" / ws.parent.name / ws.name
    d = root / "shared"
    d.mkdir(parents=True, exist_ok=True)
    return d / "autotune_configs.json"


def tune_shape(n_w: int, k: int, max_tokens: int) -> None:
    """Tune one weight shape the way a server would.

    No `tuning_buckets`: the server does not pass any either -- it runs one
    forward at max_num_batched_tokens and lets FlashInfer pick the grid. Any
    divergence here shows up as keys the server never looks up.
    """
    import torch
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    from flashinfer.autotuner import autotune

    dt = torch.bfloat16

    def q(t):
        gs = FP4_MAX / t.float().abs().max()
        f, s = nvfp4_quantize(t, gs, sfLayout=SfLayout.layout_128x4,
                              do_shuffle=False)
        return f, s, gs

    w = torch.randn(n_w, k, device="cuda", dtype=dt) * 0.05
    wf, wsf, wg = q(w)
    a = torch.randn(max_tokens, k, device="cuda", dtype=dt) * 0.05
    af, asf, ag = q(a)
    # 0-dim, NOT .reshape(1). A 1-element tensor changes the cache key and the
    # entry can never be hit. See the module docstring.
    alpha = (1.0 / (ag * wg)).float().reshape(())
    out = torch.empty(max_tokens, n_w, dtype=dt, device="cuda")
    with autotune(True):
        mm_fp4(af, wf.T, asf, wsf.T, alpha, dt, out, backend="cutlass")
    torch.cuda.synchronize()
    del w, wf, wsf, a, af, asf, out
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exports", nargs="+",
                    help="exported model directories to cover")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="max_num_batched_tokens the servers will run with; "
                         "FlashInfer tunes the buckets up to this")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the shapes and exit without touching a GPU")
    args = ap.parse_args()

    shapes: set[tuple[int, int]] = set()
    per_export = {}
    for d in args.exports:
        s = shapes_from_export(d)
        if s:
            per_export[d] = len(s)
            shapes |= s
    if not shapes:
        print("no 2-D layer weights found in the given directories",
              file=sys.stderr)
        return 2

    total_pairs = sum(per_export.values())
    print(f"{len(per_export)} export(s), {total_pairs} (export, shape) pairs, "
          f"{len(shapes)} distinct shapes")
    if total_pairs:
        print(f"  sharing avoids {100 * (1 - len(shapes) / total_pairs):.0f}% "
              f"of the tuning work")
    for n_w, k in sorted(shapes):
        print(f"    n_w={n_w:<7} K={k}")
    if args.dry_run:
        return 0

    try:
        import torch
        from flashinfer.autotuner import AutoTuner
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    path = shared_cache_path()
    before = 0
    if path.exists():
        try:
            before = len([k for k in json.load(open(path)) if k != "_metadata"])
        except Exception:                                  # noqa: BLE001
            before = 0
    # Start from whatever is already there so repeated runs accumulate rather
    # than clobber a sibling's work.
    tuner = AutoTuner.get()
    if before:
        tuner.load_configs(str(path))
    print(f"\ncache: {path}  ({before} existing entries)")

    t0 = time.time()
    for i, (n_w, k) in enumerate(sorted(shapes), 1):
        print(f"  [{i}/{len(shapes)}] n_w={n_w} K={k} ...", flush=True)
        try:
            tune_shape(n_w, k, args.max_tokens)
        except Exception as e:                             # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {str(e)[:80]}")

    tuner.save_configs(str(path))
    after = len([k for k in json.load(open(path)) if k != "_metadata"])
    print(f"\nwrote {path}")
    print(f"entries {before} -> {after}  (+{after - before})  "
          f"in {time.time() - t0:.0f}s")
    if after == before:
        print("\nWARNING: nothing was added. Either every shape was already "
              "cached, or the operand ranks no longer match what the server "
              "looks up -- check tests/test_autotune_prewarm.py.",
              file=sys.stderr)
        return 1
    print("\nServers must run with RESIDUE_SHARE_AUTOTUNE_CACHE=1 "
          "(serve_vllm.py --share_autotune_cache) to read this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
