#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Static contract checks for the CuTeDSL fold kernels.

A CuTeDSL kernel is only compiled when it is first CALLED, on a Blackwell GPU.
So the usual safety net does not exist here: a caller that passes an argument
the callee no longer accepts is invisible to `py_compile`, invisible to an
import, and only explodes at runtime, on hardware, minutes into a job. These
checks recover what a compiler would normally tell you, in about a second, with
nothing but the standard library.

Checks
------
1. ctor-kwargs   Every keyword host.py hands to a kernel class exists in that
                 class's __init__. The class is chosen by a dict lookup, so the
                 name is a variable and no linter can resolve it -- resolve it
                 here instead.
2. wrapper-arity `cute.compile(gemm.wrapper, *args)` binds POSITIONALLY. A count
                 mismatch does not raise: it silently binds values to the wrong
                 parameters. Check each host branch against the wrapper it targets.
3. decorators    A dangling decorator is legal Python: it silently attaches to
                 whatever function comes next, double-decorating it.
4. names         Any name that is read but never bound anywhere in the file.

Usage:  python3 tools/check_kernel_contracts.py [package_dir]
Exit 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import ast
import builtins
import pathlib
import sys

PKG = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else "csrc/nvfp4/nvfp4/cute_fold")
KERNELS = [("kernel_sm100_fold.py", "Sm100BlockScaledPersistentDenseGemmKernel"),
           ("kernel_sm103_fold.py", "Sm103BlockScaledPersistentDenseGemmKernel")]
FILES = ["host.py", "tactics.py", "fold_epilogue.py"] + [f for f, _ in KERNELS]

failures: list[str] = []


def _report(name: str, detail: str, before: int) -> None:
    n = len(failures) - before
    print(f"  {name:<14}: {'FAIL (%d)' % n if n else 'ok'}  {detail}")


def _tree(name: str) -> ast.Module:
    return ast.parse((PKG / name).read_text())


def _method(cls_name: str, tree: ast.Module, want: str):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == cls_name:
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name == want:
                    return m
    return None


# ---------------------------------------------------------------- 1. ctor kwargs
def check_ctor_kwargs() -> None:
    n0 = len(failures)
    host = _tree("host.py")
    passed = [{kw.arg for kw in n.keywords}
              for n in ast.walk(host)
              if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "kernel_cls"]
    accepts = []
    for fname, cls in KERNELS:
        init = _method(cls, _tree(fname), "__init__")
        a = init.args
        accepts.append((cls, {x.arg for x in a.args + a.kwonlyargs} - {"self"}))
    if len(passed) != len(accepts):
        failures.append(f"ctor-kwargs: host.py has {len(passed)} kernel_cls(...) "
                        f"call sites but there are {len(accepts)} kernel classes")
        return
    for kws, (cls, params) in zip(passed, accepts):
        extra = kws - params
        if extra:
            failures.append(f"ctor-kwargs: host.py passes {sorted(extra)} to "
                            f"{cls}(...), which its __init__ does not accept")
    _report("ctor-kwargs", f"{len(passed)} call site(s)", n0)


# -------------------------------------------------------------- 2. wrapper arity
def check_wrapper_arity() -> None:
    n0 = len(failures)
    host = _tree("host.py")
    calls = [n for n in ast.walk(host)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "compile"]
    wrappers = []
    for fname, cls in KERNELS:
        w = _method(cls, _tree(fname), "wrapper")
        a = w.args
        params = [x.arg for x in a.args if x.arg != "self"]
        wrappers.append((cls, len(params) - len(a.defaults), len(params)))
    if len(calls) != len(wrappers):
        failures.append(f"wrapper-arity: {len(calls)} cute.compile call(s) but "
                        f"{len(wrappers)} wrappers")
        return
    for c, (cls, lo, hi) in zip(calls, wrappers):
        npos = len(c.args) - 1          # minus the wrapper itself
        if not lo <= npos <= hi:
            failures.append(f"wrapper-arity: host.py passes {npos} positional "
                            f"args to {cls}.wrapper, which accepts {lo}..{hi}")
    _report("wrapper-arity", f"{len(calls)} cute.compile call(s)", n0)


# ----------------------------------------------------------------- 3. decorators
def check_decorators() -> None:
    n0 = len(failures)
    n_dec = 0
    for f in FILES:
        src = (PKG / f).read_text()
        lines = src.split("\n")
        legit: set[int] = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in n.decorator_list:
                    legit.update(range(d.lineno, d.end_lineno + 1))
                    n_dec += 1
                names = [getattr(d, "attr", getattr(d, "id", "?"))
                         for d in n.decorator_list]
                if len(names) != len(set(names)):
                    failures.append(f"decorators: {f}:{n.lineno} {n.name} is "
                                    f"decorated twice with the same decorator "
                                    f"{names}")
        for i, ln in enumerate(lines, 1):
            if ln.lstrip().startswith("@") and i not in legit:
                failures.append(f"decorators: {f}:{i} dangling decorator "
                                f"'{ln.strip()}' -- it will attach to the next "
                                f"function")
    _report("decorators", f"{n_dec} decorator(s)", n0)


# ---------------------------------------------------------------------- 4. names
def check_names() -> None:
    n0 = len(failures)
    B = set(dir(builtins))
    for f in FILES:
        tree = ast.parse((PKG / f).read_text())
        bound, loads = set(), {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, (ast.Store, ast.Del)):
                    bound.add(n.id)
                else:
                    loads.setdefault(n.id, n.lineno)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(n.name)
                a = getattr(n, "args", None)
                if a:
                    for x in a.args + a.kwonlyargs + a.posonlyargs:
                        bound.add(x.arg)
                    for x in (a.vararg, a.kwarg):
                        if x:
                            bound.add(x.arg)
            elif isinstance(n, ast.Lambda):
                for x in n.args.args:
                    bound.add(x.arg)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    bound.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
            elif isinstance(n, ast.Global):
                bound.update(n.names)
        missing = {k: v for k, v in loads.items() if k not in bound and k not in B}
        for k, ln in sorted(missing.items(), key=lambda kv: kv[1]):
            failures.append(f"names: {f}:{ln} '{k}' is read but never bound")
    extra = _serving_files()
    for path in extra:
        tree = ast.parse(path.read_text())
        bound, loads = _bindings(tree)
        for k, ln in sorted(
                ((k, v) for k, v in loads.items()
                 if k not in bound and k not in B), key=lambda kv: kv[1]):
            failures.append(f"names: {path.name}:{ln} '{k}' is read but never bound")
    _report("names", f"{len(FILES) + len(extra)} file(s)", n0)


def _serving_files() -> list:
    """The serving modules, checked by the same rule as the kernels.

    A stale name here fails exactly the way it does in a kernel: inside a
    best-effort `except Exception` that prints one line and carries on. That is
    how `mext_prefill_folds` -- deleted with the fold-at-prefill route and left
    behind at one use site -- disabled the mext_r1 kloop pre-compile for days
    while serving quietly JIT-compiled inside cudagraph capture instead. One
    second of AST here would have caught it the day it landed.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "quant"
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def _bindings(tree):
    """Names bound anywhere in the module, and names read, with first line."""
    bound, loads = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)
            else:
                loads.setdefault(n.id, n.lineno)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
            a = getattr(n, "args", None)
            if a:
                for x in a.args + a.kwonlyargs + a.posonlyargs:
                    bound.add(x.arg)
                for x in (a.vararg, a.kwarg):
                    if x:
                        bound.add(x.arg)
        elif isinstance(n, ast.Lambda):
            for x in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                bound.add(x.arg)
            for x in (n.args.vararg, n.args.kwarg):
                if x:
                    bound.add(x.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Global):
            bound.update(n.names)
        elif isinstance(n, (ast.comprehension,)):
            for tgt in ast.walk(n.target):
                if isinstance(tgt, ast.Name):
                    bound.add(tgt.id)
    return bound, loads


# ------------------------------------------------------------- 5. smem fields
def check_shared_storage() -> None:
    """Every `storage.X` read must be a field of that file's SharedStorage.

    SharedStorage is built at trace time, so a read of a field that no longer
    exists is invisible to py_compile and to an import -- it surfaces as an
    AttributeError on the GPU, minutes into a job. Deleting a field while
    leaving a reader behind has happened twice; this makes it a one-second
    check instead of a GPU round trip.
    """
    n0 = len(failures)
    n_field = 0
    for fname, _cls in KERNELS:
        tree = _tree(fname)
        fields = {st.target.id
                  for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "SharedStorage"
                  for st in n.body
                  if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)}
        reads = {n.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                 and n.value.id == "storage"}
        n_field += len(fields)
        for miss in sorted(reads - fields):
            failures.append(f"shared-storage: {fname} reads storage.{miss}, "
                            f"which SharedStorage does not define")
    _report("shared-storage", f"{n_field} field(s)", n0)


# ------------------------------------------------------- 6. runtime call arity
# REMOVED: compiled-arity.
#
# It asserted that every `compiled(...)` call site in host.py passes the same
# number of runtime arguments. That held while one wrapper shape existed. It no
# longer does: the host now dispatches four shapes whose dynamic-argument counts
# legitimately differ -- M-ext (sm100), fold_tma with a split-K workspace, the
# sm103 branch without it, and the wrapped-K-loop entry point. Every run of this
# check therefore failed on correct code, which is worse than no check: a red CI
# that is always wrong teaches people to stop reading it.
#
# What the removal does NOT give up: the dangerous case is `cute.compile(
# gemm.wrapper, *args)`, which binds POSITIONALLY and silently binds values to
# the wrong parameters on a count mismatch. That is still covered, per-arch, by
# wrapper-arity above. A mismatch on the runtime `compiled(...)` call raises
# TypeError instead -- late and on a GPU, but loud, and never silently wrong.


def main() -> int:
    if not PKG.is_dir():
        print(f"package dir not found: {PKG}", file=sys.stderr)
        return 1
    print(f"kernel contract checks on {PKG}")
    check_ctor_kwargs()
    check_wrapper_arity()
    check_decorators()
    check_names()
    check_shared_storage()
    if failures:
        print(f"\nFAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nOK -- all contracts hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
