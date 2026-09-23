# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Scoped research adapter for CuTe 4.6 + an external CUDA 13.4 ptxas.

The frontend emits real QMUL4 PTX, never FMUL placeholders. Only the PTX
version directive is raised to 9.4. The assembled ELF replaces the loader's
PTX payload before TVM finalization; no SASS instructions are rewritten.
This does not replace the process's CUDA toolkit or affect other compiles.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import os
import re
import subprocess
import tempfile


_native_ptxas = ContextVar("resattn_native_ptxas", default=None)


def native_qmul4_enabled():
    return _native_ptxas.get() is not None


def _compile_native(original, compiler, module, pipeline, *args, **kwargs):
    from cutlass._mlir import ir

    if "cubin-chip=sm_100a" not in pipeline:
        raise ValueError("external QMUL4 compiler is validated only for sm_100a")
    ptxas = _native_ptxas.get()
    dump_root = os.environ.get("RESATTN_NATIVE_DUMP_DIR")
    if dump_root:
        Path(dump_root).mkdir(parents=True, exist_ok=True)
    # Retain compiler evidence even when compilation fails.
    directory = Path(tempfile.mkdtemp(prefix="resattn-native-", dir=dump_root))
    (directory / "pipeline.txt").write_text(pipeline)
    hook = compiler._post_compile_hook
    compiler._post_compile_hook = None
    try:
        stage1 = pipeline.replace(
            "cubin-format=bin", "cubin-format=bin llvm-only=true check-inline-asm=false"
        )
        original(compiler, module, stage1, *args, **kwargs)
        gpu_modules = 0

        def set_ptx_target(op):
            nonlocal gpu_modules
            if str(op.name) == "gpu.module":
                gpu_modules += 1
                if "targets" in op.attributes:
                    del op.attributes["targets"]
                op.attributes["compute_targets"] = ir.ArrayAttr.get([
                    ir.Attribute.parse("#cuda.compute_target<ptx, conditional, [sm_100]>")
                ])
            return ir.WalkResult.ADVANCE

        module.operation.walk(set_ptx_target, ir.WalkOrder.PRE_ORDER)
        if gpu_modules != 1:
            raise RuntimeError(f"expected one GPU module, got {gpu_modules}")
        source = directory / "kernel.ptx"
        original(
            compiler, module,
            "builtin.module(cuda-to-binary{opt-level=3 dump-ptx-path="
            + str(source) + "},cuda-to-llvm,cuda-error-handling,convert-scf-to-cf,convert-cf-to-llvm,canonicalize,cse,reconcile-unrealized-casts)",
            *args, **kwargs,
        )
        # cuda-to-binary appends the concrete architecture and extension.
        ptx = Path(str(source) + ".sm_100a.ptx").read_text()
        if "mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4" not in ptx:
            raise RuntimeError("native compile did not emit the padded QMUL4 instruction")
        ptx, count = re.subn(r"(?m)^\.version\s+9\.[34]\s*$", ".version 9.4", ptx)
        if count != 1:
            raise RuntimeError("unexpected PTX version directive")
        source.write_text(ptx)
        cubin = directory / "kernel.cubin"
        command = [ptxas, "-arch=sm_100a", "-O3", "-v", str(source), "-o", str(cubin)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (directory / "ptxas.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"external ptxas failed: {directory}\n{result.stderr}")
        blob = cubin.read_bytes()
        if not blob.startswith(b"\x7fELF"):
            raise RuntimeError("ptxas did not produce an ELF cubin")
        replaced = 0

        def replace_binary(op):
            nonlocal replaced
            if (str(op.name) == "llvm.mlir.global" and "sym_name" in op.attributes
                    and ir.StringAttr(op.attributes["sym_name"]).value == "kernels_binary"):
                op.attributes["value"] = ir.StringAttr.get(blob)
                op.attributes["global_type"] = ir.TypeAttr.get(
                    ir.Type.parse(f"!llvm.array<{len(blob)} x i8>")
                )
                replaced += 1
            return ir.WalkResult.ADVANCE

        module.operation.walk(replace_binary, ir.WalkOrder.PRE_ORDER)
        if replaced != 1:
            raise RuntimeError(f"expected one kernels_binary, got {replaced}")
    finally:
        compiler._post_compile_hook = hook
    if hook:
        hook(module)
    (directory / "host.mlir").write_text(str(module))


@contextmanager
def native_qmul4_compilation(ptxas):
    """Select a separate, explicit native specialization; restore on failure."""
    from cutlass.base_dsl.compiler import Compiler

    path = Path(ptxas).resolve(strict=True)
    version = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=True)
    if not re.search(r"release 13\.4\b", version.stdout):
        raise ValueError("the research adapter currently requires CUDA 13.4 ptxas")
    if not getattr(Compiler, "_resattn_external_ptxas", False):
        original = Compiler.compile

        @wraps(original)
        def compile_scoped(self, module, pipeline, *args, **kwargs):
            if native_qmul4_enabled():
                return _compile_native(original, self, module, pipeline, *args, **kwargs)
            return original(self, module, pipeline, *args, **kwargs)

        Compiler.compile = compile_scoped
        Compiler._resattn_external_ptxas = True
    token = _native_ptxas.set(str(path))
    try:
        yield
    finally:
        _native_ptxas.reset(token)
