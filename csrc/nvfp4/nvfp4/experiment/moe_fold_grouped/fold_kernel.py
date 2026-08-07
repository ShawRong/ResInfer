# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""An m-major-C variant of flashinfer's gather grouped MoE GEMM.

Why: `get_tmem_load_op` picks the `.16x256b` TMEM-load atom when the C layout
is m-major, and that atom co-locates M-rows {p, p+8} in ONE thread's
accumulator fragment. That co-location is the whole reason our dense M-ext
fold (A4/A6) can add a (base, residue) pair in registers. Their kernel and
ours make the byte-identical `get_tmem_load_op` call; the only difference is
that ours pins COL_MAJOR and theirs derives the layout -- and derives it from
a layout it builds itself:

    c = cute.make_tensor(
        c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2)))

`order=(1,0,2)` puts N innermost, so it is always ROW_MAJOR and passing an
m-major torch tensor changes nothing (the wrapper only reads `data_ptr()`).

The fork is that one tuple.

How the fork is taken
---------------------
NOT by copying the ~90-line `wrapper` method into this file. It is derived
from the parent's own source at import time, with the substitution asserted to
match exactly once. If flashinfer changes that line, or changes it in a way
that no longer matches, this raises at import instead of silently reverting to
row-major -- which would look like "the fold just doesn't help" rather than
"the fold never ran". Same discipline as csrc/nvfp4/nvfp4/cute_fold/FORK.md,
applied to a one-line fork.

Cost of m-major C: the OUTPUT is then written m-major, so callers must read it
with that layout. Whether the rest of the epilogue tolerates it is exactly what
`probe_m_major.py` measures -- `can_implement(..., c_major="n")` upstream
suggests the n-major assumption may appear elsewhere too.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import textwrap

_ORDER_ROW_MAJOR = "layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))"
_ORDER_COL_MAJOR = "layout=cute.make_ordered_layout((m, interm_size, 1), order=(0, 1, 2))"

_cls_cache = {}


def m_major_kernel_class():
    """Return (and cache) the m-major subclass, derived from upstream source."""
    if "cls" in _cls_cache:
        return _cls_cache["cls"]

    from flashinfer.fused_moe.cute_dsl.blackwell import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion as up,
    )

    parent = up.BlockScaledContiguousGatherGroupedGemmKernel
    src = textwrap.dedent(inspect.getsource(parent.wrapper))

    n = src.count(_ORDER_ROW_MAJOR)
    if n != 1:
        raise RuntimeError(
            "moe_fold_grouped: expected exactly one C-layout site in "
            f"{parent.__name__}.wrapper, found {n}. flashinfer changed the "
            "kernel -- re-read wrapper() and update _ORDER_ROW_MAJOR rather "
            "than loosening this check. Silently keeping row-major would make "
            "the fold look useless instead of absent."
        )

    src = src.replace(_ORDER_ROW_MAJOR, _ORDER_COL_MAJOR)

    # inspect.getsource follows __wrapped__ and hands back the UNDECORATED
    # body, so @cute.jit is missing and cute.compile() rejects the result with
    # CALL_MISSING_JIT_DECORATOR. Put it back rather than assuming getsource
    # kept it.
    if not src.lstrip().startswith("@"):
        src = "@cute.jit\n" + src

    # The derived source must live in a REAL file. cute.jit's AST
    # preprocessor re-reads the function's source via inspect, so a synthetic
    # co_filename makes it fail with "Failed to parse function wrapper".
    # Globals still come from the upstream module, so cute/cutlass/@cute.jit
    # resolve exactly as they do there.
    tmpdir = tempfile.mkdtemp(prefix="moe_fold_derived_")
    path = os.path.join(tmpdir, "derived_wrapper.py")
    with open(path, "w") as fh:
        fh.write(src)
    _cls_cache["src_path"] = path

    ns = dict(up.__dict__)
    exec(compile(src, path, "exec"), ns)  # noqa: S102

    cls = type(
        "MMajorGatherGroupedGemmKernel", (parent,), {"wrapper": ns["wrapper"]}
    )
    cls.__doc__ = (
        "flashinfer's gather grouped MoE GEMM with the C layout flipped to "
        "m-major, so get_tmem_load_op selects the .16x256b atom. Derived from "
        "upstream source at import; see fold_kernel.py."
    )
    _cls_cache["cls"] = cls
    return cls


def upstream_kernel_class():
    from flashinfer.fused_moe.cute_dsl.blackwell import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion as up,
    )

    return up.BlockScaledContiguousGatherGroupedGemmKernel


def describe_fork() -> str:
    """One-line provenance, for probe output so a result names its variant."""
    import flashinfer

    return (f"flashinfer {getattr(flashinfer, '__version__', '?')}, "
            f"C layout order (1,0,2)->(0,1,2) [1 site]")
