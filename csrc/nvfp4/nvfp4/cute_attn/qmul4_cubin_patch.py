# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Temporary CuTe DSL bridge for the packed NVFP4 multiply on SM100.'

from __future__ import annotations

import os
import struct
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Callable


_FMUL_OPCODE = 0x7220
# ptxas is allowed to uniformize the packed scale operand.  The atomic
# reduction variant of the decode kernel does so for half of the conversion
# markers, spelling them as ``FMUL.RZ Rd, Ra, URb`` instead of using a GPR.
# CUDA 13.4 ptxas proves that the corresponding public QMUL4 encoding uses a
# distinct opcode skeleton plus bit 27 in the format word:
#
#   QMUL4 ... R5, R0.N3210, UR7.B3210
#   lo = 0x0000000700057c7c, hi.low32 = 0x0d01b800
_FMUL_UNIFORM_OPCODE = 0x7C20






_QMUL4_LO_SKELETON = 0x000000000000727C
_QMUL4_UNIFORM_LO_SKELETON = 0x0000000000007C7C
_REGISTER_BYTES_MASK = 0x000000FFFFFF0000
_QMUL4_FORMAT_N3210 = 0x0501B800
_QMUL4_FORMAT_N7654 = 0x0501B400
_QMUL4_FORMAT_N6420 = 0x0501BC00
_QMUL4_UNIFORM_SOURCE2 = 0x08000000
_ROUND_RM = 1
_ROUND_RP = 2
_ROUND_RZ = 3
_SOURCE_REGISTER_BYTES_MASK = 0x000000FFFF000000

_PATCH_ENV = "RESATTN_QMUL4_PATCH"
_N6420_PATCH_ENV = "RESATTN_QMUL4_N6420_PATCH"
_DUMP_ENV = "RESATTN_QMUL4_DUMP_DIR"
_LOADER_HOOK_SENTINEL = "_resattn_qmul4_loader_patch"
_COMPILER_HOOK_SENTINEL = "_resattn_qmul4_compile_patch"
_SUPPORTED_SM = (100, 103)
_dump_sequence = count()


class _Qmul4MarkerError(ValueError):
    """A real cubin contained an N6420 marker that is unsafe to rewrite."""


def _elf_text_sections(blob: bytes | bytearray) -> list[tuple[int, int]]:
    """Return ``(file_offset, size)`` for every ELF ``.text.*`` section."""
    if len(blob) < 64 or blob[:4] != b"\x7fELF":
        raise ValueError("QMUL4 patch input is not an ELF cubin")
    if blob[4] != 2 or blob[5] != 1:
        raise ValueError("QMUL4 patch supports only little-endian ELF64 cubins")

    section_table = struct.unpack_from("<Q", blob, 0x28)[0]
    section_entry_size = struct.unpack_from("<H", blob, 0x3A)[0]
    section_count = struct.unpack_from("<H", blob, 0x3C)[0]
    string_table_index = struct.unpack_from("<H", blob, 0x3E)[0]
    table_end = section_table + section_count * section_entry_size
    if section_entry_size < 64 or table_end > len(blob):
        raise ValueError("invalid ELF section table in QMUL4 cubin")
    if string_table_index >= section_count:
        raise ValueError("invalid ELF section-name table index")

    string_header = section_table + string_table_index * section_entry_size
    string_offset = struct.unpack_from("<Q", blob, string_header + 0x18)[0]
    string_size = struct.unpack_from("<Q", blob, string_header + 0x20)[0]
    string_end = string_offset + string_size
    if string_end > len(blob):
        raise ValueError("invalid ELF section-name table")

    sections: list[tuple[int, int]] = []
    for index in range(section_count):
        header = section_table + index * section_entry_size
        name_offset = struct.unpack_from("<I", blob, header)[0]
        if name_offset >= string_size:
            continue
        name_begin = string_offset + name_offset
        name_end = blob.find(b"\0", name_begin, string_end)
        if name_end < 0:
            continue
        if blob[name_begin:name_end].startswith(b".text."):
            offset = struct.unpack_from("<Q", blob, header + 0x18)[0]
            size = struct.unpack_from("<Q", blob, header + 0x20)[0]
            if offset + size > len(blob):
                raise ValueError("invalid ELF text section")
            sections.append((offset, size))
    return sections


def patch_qmul4_placeholders(cubin: bytes | bytearray | memoryview) -> tuple[bytes, int]:
    'Replace marked FMUL placeholders and return ``(patched_cubin, count)``.'
    blob = bytearray(cubin)
    if len(blob) < 64 or blob[:4] != b"\x7fELF":
        raise ValueError("QMUL4 patch input is not an ELF cubin")
    # CUDA stores the target SM in bits 8..15 of ELF e_flags (0x64=SM100,
    # 0x67=SM103).  The opcode is architecture-specific; silently leave all
    # other cubins alone because the loader hook is process-wide.
    sm_arch = (struct.unpack_from("<I", blob, 0x30)[0] >> 8) & 0xFF
    if sm_arch not in _SUPPORTED_SM:
        return bytes(blob), 0
    patched = 0
    for section_offset, section_size in _elf_text_sections(blob):
        section_end = section_offset + section_size
        offset = section_offset
        while offset + 15 < section_end:
            lo, hi = struct.unpack_from("<QQ", blob, offset)
            rounding = (hi >> 14) & 3
            accepted_roundings = (_ROUND_RM, _ROUND_RP)
            if os.environ.get(_N6420_PATCH_ENV) == "1":
                # Experimental TRT-LLM decode bridge.  LDSM.U4 expands four
                # FP4 nibbles into the low nibble of four byte containers;
                # TRT consumes that register directly with ``.N6420`` and a
                # packed four-byte scale operand (``.B3210``).  PTX 9.3 has
                # no spelling for that source selector, so an opt-in FMUL.RZ
                # marker carries it until the cubin is materialized.
                #
                # Keep this opt-in: the loader hook is process-wide, whereas
                # RM/RP were proven absent from the native decode cubin.  We
                # do not yet make that stronger claim for every RZ user that
                # could share a process with the experiment.
                accepted_roundings += (_ROUND_RZ,)
            opcode = lo & 0xFFFF
            uniform_src2 = opcode == _FMUL_UNIFORM_OPCODE
            regular_src2 = opcode == _FMUL_OPCODE
            # Public PTX 9.4 gives ground truth for N3210/N7654 with a UR
            # scale, so RM/RP uniform markers are legal.  The analogous
            # N6420+UR bitwise composition is *not* legal: B200 Q0 drops from
            # 0.9997344 to -0.0317149.  qmul4.py therefore materializes its
            # scale in a GPR.  Fail loudly if ptxas ever undoes that contract
            # instead of silently installing the disproved encoding again.
            if (
                uniform_src2
                and rounding == _ROUND_RZ
                and os.environ.get(_N6420_PATCH_ENV) == "1"
            ):
                raise _Qmul4MarkerError(
                    "N6420 QMUL4 marker was uniformized; the direct decode "
                    "path requires a GPR scale operand"
                )
            accepted_opcode = regular_src2 or (
                uniform_src2 and rounding in (_ROUND_RM, _ROUND_RP)
            )
            if not accepted_opcode or rounding not in accepted_roundings:
                offset += 16
                continue
            if rounding == _ROUND_RM:
                qmul_format = _QMUL4_FORMAT_N3210
            elif rounding == _ROUND_RP:
                qmul_format = _QMUL4_FORMAT_N7654
            else:
                qmul_format = _QMUL4_FORMAT_N6420
            qmul_skeleton = _QMUL4_LO_SKELETON
            if uniform_src2:
                qmul_skeleton = _QMUL4_UNIFORM_LO_SKELETON
                qmul_format |= _QMUL4_UNIFORM_SOURCE2
            qmul_lo = (
                (qmul_skeleton & ~_REGISTER_BYTES_MASK)
                | (lo & _REGISTER_BYTES_MASK)
            )
            qmul_hi = (hi & 0xFFFFFFFF00000000) | qmul_format
            struct.pack_into("<QQ", blob, offset, qmul_lo, qmul_hi)
            patched += 1
            offset += 16
    return bytes(blob), patched


def patch_qmul4_container(
    container: bytes | bytearray | memoryview,
) -> tuple[bytes, int]:
    """Patch every uncompressed cubin embedded in an ELF or CUDA fatbinary.

    CuTe's ``cute-to-nvvm`` pass embeds a CUDA fatbinary as an LLVM string
    global.  The cubin itself remains an ordinary ELF image inside that
    container, so its section offsets and instruction scheduling words can be
    patched without changing the global's size.
    """
    blob = bytes(container)
    patched_total = 0
    search_from = 0
    while True:
        elf_offset = blob.find(b"\x7fELF", search_from)
        if elf_offset < 0:
            break
        try:
            patched_suffix, patched = patch_qmul4_placeholders(blob[elf_offset:])
        except _Qmul4MarkerError:
            # This was a valid target cubin with a marker that would become a
            # semantically wrong QMUL4.  Never downgrade it to a coincidental
            # ELF match and silently continue loading the placeholder FMUL.
            raise
        except ValueError:
            # A coincidental ELF magic in another payload is not a cubin.
            search_from = elf_offset + 4
            continue
        if patched:
            blob = blob[:elf_offset] + patched_suffix
            patched_total += patched
        search_from = elf_offset + 4
    return blob, patched_total


def patch_qmul4_cubin_file(source: str | os.PathLike, output: str | os.PathLike) -> int:
    """Patch one cubin file, primarily for offline inspection."""
    patched, count = patch_qmul4_placeholders(Path(source).read_bytes())
    Path(output).write_bytes(patched)
    return count


def _patch_enabled() -> bool:
    return os.environ.get(_PATCH_ENV, "1") != "0"


def _patch_and_maybe_dump(data, suffix: str) -> bytes:
    blob = bytes(data)
    if not _patch_enabled():
        return blob
    patched, patched_count = patch_qmul4_container(blob)
    if patched_count:
        dump_dir = os.environ.get(_DUMP_ENV)
        if dump_dir:
            path = Path(dump_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"qmul4_{next(_dump_sequence)}{suffix}").write_bytes(patched)
    return patched


def _patch_embedded_cubins(module) -> int:
    """Rewrite CuTe's post-lowering ``kernels_binary`` LLVM global in place."""
    from cutlass._mlir import ir

    patched_total = 0

    def visit(op):
        nonlocal patched_total
        if str(op.name) != "llvm.mlir.global":
            return ir.WalkResult.ADVANCE
        if "sym_name" not in op.attributes or "value" not in op.attributes:
            return ir.WalkResult.ADVANCE
        sym_name = ir.StringAttr(op.attributes["sym_name"]).value
        if sym_name != "kernels_binary":
            return ir.WalkResult.ADVANCE
        value = ir.StringAttr(op.attributes["value"]).value_bytes
        patched, patched_count = patch_qmul4_container(value)
        if patched_count:
            # Instruction substitution is size preserving, so the LLVM array
            # type and every host-side address calculation remain unchanged.
            op.attributes["value"] = ir.StringAttr.get(patched)
            patched_total += patched_count
            dump_dir = os.environ.get(_DUMP_ENV)
            if dump_dir:
                path = Path(dump_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / f"qmul4_{next(_dump_sequence)}.fatbin").write_bytes(patched)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return patched_total


def install_qmul4_loader_patch() -> None:
    """Install idempotent CuTe compile/load hooks for the CUDA 13.3 fallback."""
    from cutlass.base_dsl.compiler import Compiler
    import cutlass.base_dsl.runtime.cuda as cuda_helpers

    # The modern CUDA-dialect and TVM-FFI executors do not call Python's CUDA
    # loader helpers: cute-to-nvvm embeds the fatbinary directly in the host
    # LLVM module.  Patch that global after the lowering pass returns but
    # before Compiler.compile_and_jit constructs the execution engine.
    if not getattr(Compiler, _COMPILER_HOOK_SENTINEL, False):
        original_compile: Callable = Compiler.compile

        @wraps(original_compile)
        def compile_with_qmul4_patch(self, module, pipeline, *args, **kwargs):
            result = original_compile(self, module, pipeline, *args, **kwargs)
            if _patch_enabled():
                _patch_embedded_cubins(module)
            return result

        Compiler.compile = compile_with_qmul4_patch
        setattr(Compiler, _COMPILER_HOOK_SENTINEL, True)

    # Keep the loader hooks for exported cubins and older CuTe executors.
    if not getattr(cuda_helpers, _LOADER_HOOK_SENTINEL, False):
        original_load_data: Callable = cuda_helpers.load_cubin_module_data
        original_load_file: Callable = cuda_helpers.load_cubin_module
        original_load_library_data: Callable = cuda_helpers.load_library_data

        def load_data(data):
            return original_load_data(_patch_and_maybe_dump(data, ".cubin"))

        def load_file(path):
            if _patch_enabled():
                data = _patch_and_maybe_dump(Path(path).read_bytes(), ".cubin")
                return original_load_data(data)
            return original_load_file(path)

        def load_library_data(data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                data = _patch_and_maybe_dump(data, ".fatbin")
            return original_load_library_data(data)

        cuda_helpers.load_cubin_module_data = load_data
        cuda_helpers.load_cubin_module = load_file
        cuda_helpers.load_library_data = load_library_data
        setattr(cuda_helpers, _LOADER_HOOK_SENTINEL, True)


__all__ = [
    "install_qmul4_loader_patch",
    "patch_qmul4_container",
    "patch_qmul4_cubin_file",
    "patch_qmul4_placeholders",
]
