#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit autotune end-to-end: is every GEMM the server will run actually tuned?

WHY

There are TWO autotune layers and each can silently under-cover:

  1. FlashInfer's fp4_gemm (every linear layer, incl. K-ext). Its cache is a
     JSON file keyed by (op, runner, full operand shapes). A prewarm whose
     operand ranks drift writes keys the server never hits; an arm that never
     finished warmup leaves shapes untuned. Either way the only live symptom
     is "startup is still slow" / "serving on fallback tactics".
  2. The CuTeDSL sm10x fold (mext_r1 layers, which fold at EVERY M). Its
     tuner caches per (family, shape, m-bucket) IN PROCESS -- historically
     invisible. RESIDUE_FOLD_TUNE_DUMP=<path> now makes the server write it
     out, and this script reads that dump.

This tool turns both into checkable files:

  coverage CACHE --exports DIR...   every export weight shape has fp4_gemm
                                    entries, and the bucket set is uniform
  diff A B                          kernel-level tune vs system e2e tune:
                                    key overlap, metadata drift, tactic
                                    agreement (disagreement on near-tied
                                    tactics is normal; missing keys are not)
  fold DUMP [--expect-buckets ...]  the fold tuner covered the buckets the
                                    serving path will look up

Exit code 0 = no gaps, 1 = gaps found, 2 = usage/IO error.

Usage:
    python tools/audit_autotune_coverage.py coverage \
        ~/.cache/vllm/flashinfer_autotune_cache/<ver>/<arch>/shared/autotune_configs.json \
        --exports /path/to/exports/*/exported_residue
    python tools/audit_autotune_coverage.py diff prewarm.json server.json
    python tools/audit_autotune_coverage.py fold fold_dump.json \
        --expect-buckets 128,256,512,1024
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_cache(path: str):
    """(metadata, {parsed_key_tuple: (runner, tactic)}) from a FlashInfer
    autotune cache JSON. Keys are repr()-ed tuples; ast.literal_eval
    round-trips them (same convention tests/test_autotune_prewarm.py pins)."""
    with open(path) as f:
        raw = json.load(f)
    meta = raw.pop("_metadata", None)
    entries = {}
    for k, v in raw.items():
        entries[ast.literal_eval(k)] = tuple(v)
    return meta, entries


def fp4_gemm_view(entries) -> dict:
    """{(n_w, k_logical): {m_bucket: tactic}} for the fp4_gemm op.

    Operand 0 is the activation (m_bucket, k_packed); operand 1 the transposed
    weight (k_packed, n_w). k_logical = 2 * k_packed (two FP4 per byte).
    """
    out: dict = {}
    for key, v in entries.items():
        op, _runner, shapes = key[0], key[1], key[2]
        if op != "fp4_gemm":
            continue
        (m, k_packed), (_k2, n_w) = shapes[0], shapes[1]
        out.setdefault((int(n_w), int(k_packed) * 2), {})[int(m)] = v
    return out


def ops_summary(entries) -> dict:
    """{op_name: entry_count} -- shows non-fp4_gemm ops (e.g. the sm120 fold
    op mext_fold_fp4_gemm) that rode the same cache."""
    out: dict = {}
    for key in entries:
        out[key[0]] = out.get(key[0], 0) + 1
    return out


def cmd_coverage(args) -> int:
    sys.path.insert(0, _HERE)
    from prewarm_autotune_cache import shapes_from_export

    meta, entries = load_cache(args.cache)
    view = fp4_gemm_view(entries)
    print(f"cache: {args.cache}")
    if meta:
        print(f"  env : {meta.get('gpu')} / flashinfer "
              f"{meta.get('flashinfer_version')} / cuda {meta.get('cuda_version')}")
    for op, n in sorted(ops_summary(entries).items()):
        print(f"  op  : {op:<24} {n} entries")

    shapes: set = set()
    for d in args.exports:
        shapes |= shapes_from_export(d)
    if not shapes:
        print("no 2-D layer weights found in the given exports", file=sys.stderr)
        return 2

    # The tuner writes one entry per bucket; every fully-tuned shape gets the
    # SAME bucket set. The largest observed set is the reference -- a shape
    # below it was interrupted or never reached.
    full = set()
    for buckets in view.values():
        if len(buckets) > len(full):
            full = set(buckets)

    absent, partial, ok = [], [], []
    for shape in sorted(shapes):
        got = view.get(shape)
        if not got:
            absent.append(shape)
        elif set(got) < full:
            partial.append((shape, sorted(full - set(got))))
        else:
            ok.append(shape)

    print(f"\nexport shapes: {len(shapes)}  "
          f"(reference bucket set: {len(full)} buckets)")
    print(f"  fully tuned : {len(ok)}")
    for shape, missing in partial:
        print(f"  PARTIAL     : n_w={shape[0]} K={shape[1]} "
              f"missing buckets {missing}")
    for shape in absent:
        print(f"  ABSENT      : n_w={shape[0]} K={shape[1]} "
              f"-- server will tune inline or serve fallback tactics")
    gaps = len(absent) + len(partial)
    print(f"\nverdict: {'OK -- every shape fully tuned' if not gaps else f'{gaps} shape(s) under-covered'}")
    return 1 if gaps else 0


def cmd_diff(args) -> int:
    meta_a, a = load_cache(args.a)
    meta_b, b = load_cache(args.b)

    drift = []
    if meta_a and meta_b:
        for field in sorted(set(meta_a) | set(meta_b)):
            if meta_a.get(field) != meta_b.get(field):
                drift.append((field, meta_a.get(field), meta_b.get(field)))
    if drift:
        print("METADATA DRIFT -- load_configs would reject one of these "
              "wholesale on the other's machine:")
        for field, va, vb in drift:
            print(f"  {field}: {va!r} vs {vb!r}")
    else:
        print("metadata: identical (same GPU / library stack)")

    only_a = sorted(set(a) - set(b), key=repr)
    only_b = sorted(set(b) - set(a), key=repr)
    common = set(a) & set(b)
    agree = [k for k in common if a[k] == b[k]]
    disagree = sorted((k for k in common if a[k] != b[k]), key=repr)

    print(f"\nA: {args.a}  ({len(a)} entries)")
    print(f"B: {args.b}  ({len(b)} entries)")
    print(f"  common keys      : {len(common)}")
    if common:
        print(f"  identical tactic : {len(agree)} "
              f"({100 * len(agree) / len(common):.0f}%)")
    print(f"  only in A        : {len(only_a)}")
    print(f"  only in B        : {len(only_b)}")

    def _describe(key):
        op, _r, shapes = key[0], key[1], key[2]
        if op == "fp4_gemm":
            (m, kp), (_k2, n) = shapes[0], shapes[1]
            return f"{op} n_w={n} K={kp * 2} m={m}"
        return f"{op} {shapes!r}"[:90]

    for label, keys in (("only in A", only_a), ("only in B", only_b)):
        for k in keys[: args.max_examples]:
            print(f"    [{label}] {_describe(k)}")
        if len(keys) > args.max_examples:
            print(f"    [{label}] ... {len(keys) - args.max_examples} more")

    if disagree:
        print(f"\n  tactic disagreements ({len(disagree)}; near-tied tactics "
              "flipping between runs is expected, missing keys are not):")
        for k in disagree[: args.max_examples]:
            print(f"    {_describe(k)}: A={a[k]} B={b[k]}")

    # Keys present in only one file mean one side would tune inline what the
    # other has cached -- that is the actionable gap; tactic flips are not.
    return 1 if (only_a or only_b) else 0


def cmd_fold(args) -> int:
    with open(args.dump) as f:
        raw = json.load(f)
    tactics = {ast.literal_eval(k): ast.literal_eval(v)
               for k, v in raw.get("tactics", {}).items()}
    families = {ast.literal_eval(k): ast.literal_eval(v)
                for k, v in raw.get("families", {}).items()}

    # tactics key: (family, arch, n_w, k_logical, m_bucket)
    by_shape: dict = {}
    for (family, arch, n_w, k, bucket), tac in tactics.items():
        by_shape.setdefault((arch, n_w, k), {}).setdefault(
            tuple(family), set()).add(bucket)

    print(f"fold dump: {args.dump}")
    print(f"  tuned (family, shape, bucket) entries : {len(tactics)}")
    print(f"  cross-family winners cached           : {len(families)}")

    expected = (set(int(b) for b in args.expect_buckets.split(","))
                if args.expect_buckets else None)
    gaps = 0
    for (arch, n_w, k), fams in sorted(by_shape.items()):
        for fam, buckets in sorted(fams.items()):
            line = (f"  {arch} n_w={n_w} K={k} family={fam}: "
                    f"buckets {sorted(buckets)}")
            if expected and not expected <= buckets:
                line += f"  MISSING {sorted(expected - buckets)}"
                gaps += 1
            print(line)
    if expected:
        print(f"\nverdict: {'OK -- all expected buckets tuned' if not gaps else f'{gaps} (shape, family) under-covered'}")
    return 1 if gaps else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coverage", help="export shapes vs one cache file")
    c.add_argument("cache")
    c.add_argument("--exports", nargs="+", required=True)
    c.set_defaults(fn=cmd_coverage)

    d = sub.add_parser("diff", help="kernel-level tune vs system e2e tune")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--max-examples", type=int, default=8)
    d.set_defaults(fn=cmd_diff)

    f = sub.add_parser("fold", help="audit a RESIDUE_FOLD_TUNE_DUMP file")
    f.add_argument("dump")
    f.add_argument("--expect-buckets", default="",
                   help="comma-separated m buckets that must be tuned, "
                        "e.g. 128,256,512,1024")
    f.set_defaults(fn=cmd_fold)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        print(f"not found: {e.filename}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
