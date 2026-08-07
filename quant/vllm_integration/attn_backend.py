# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'ResAttn: vLLM v1 attention backend for the paged NVFP4 KV cache.'

from __future__ import annotations

import functools
import itertools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

try:  # removed from vllm.utils.torch_utils in vLLM 0.26.0
    from vllm.utils.torch_utils import nvfp4_kv_cache_split_views
except ImportError:

    def _nvfp4_split_data_scale(
        kv_side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split one NVFP4 KV-side buffer into data and scale views.

        Vendored verbatim from vLLM <= 0.25 (`vllm/utils/torch_utils.py`),
        which dropped this helper in 0.26.0. Per-side physical layout is
        [data | scale]; the inner strides are derived from the input so the
        NHD/HND order of the caller is preserved.
        """
        num_pages = kv_side.shape[0]
        dim_1, dim_2 = kv_side.shape[1], kv_side.shape[2]
        full_dim = kv_side.shape[3]
        data_dim = full_dim * 8 // 9
        scale_dim = full_dim - data_dim

        data_per_kv = dim_1 * dim_2 * data_dim
        page_bytes = kv_side.stride(0)

        s1 = kv_side.stride(1) * data_dim // full_dim
        s2 = kv_side.stride(2) * data_dim // full_dim
        data_shape = (num_pages, dim_1, dim_2, data_dim)
        data_strides = (page_bytes, s1, s2, 1)

        s1_s = kv_side.stride(1) * scale_dim // full_dim
        s2_s = kv_side.stride(2) * scale_dim // full_dim
        scale_shape = (num_pages, dim_1, dim_2, scale_dim)
        scale_strides = (page_bytes, s1_s, s2_s, 1)

        base = kv_side.storage_offset()
        data = torch.as_strided(
            kv_side, data_shape, data_strides, storage_offset=base
        )
        scale = torch.as_strided(
            kv_side, scale_shape, scale_strides, storage_offset=base + data_per_kv
        ).view(torch.float8_e4m3fn)
        return data, scale

    def nvfp4_kv_cache_split_views(kv_cache: torch.Tensor) -> tuple[tuple, tuple]:
        """Split an NVFP4 KV cache into data and scale views (5D or 4D)."""
        if kv_cache.dim() == 4:
            data, scale = _nvfp4_split_data_scale(kv_cache)
            return (data,), (scale,)
        k_data, k_scale = _nvfp4_split_data_scale(kv_cache[:, 0])
        v_data, v_scale = _nvfp4_split_data_scale(kv_cache[:, 1])
        return (k_data, v_data), (k_scale, v_scale)

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec

# --------------------------------------------------------------------------- #
# Management layer (csrc/nvfp4/nvfp4/cute_attn, editable-installed as `nvfp4`;
# same import convention as residue_linear_op.py / linear_method_v2.py).
# --------------------------------------------------------------------------- #
try:
    from nvfp4.cute_attn import (  # type: ignore[import-not-found]
        PAGE_SIZE as _MMA_PAGE_SIZE,
        SF_VEC as _SF_VEC,
        NvFp4PagedKVPool,
        gather_dequant_trtllm,
    )

    _HAS_CUTE_ATTN = True
    _CUTE_ATTN_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover - environment without nvfp4 pkg
    _HAS_CUTE_ATTN = False
    _CUTE_ATTN_IMPORT_ERROR = _e
    _MMA_PAGE_SIZE = 128  # design doc D1 (frozen)
    _SF_VEC = 16
    NvFp4PagedKVPool = None  # type: ignore[assignment]
    gather_dequant_trtllm = None  # type: ignore[assignment]


















_DECODE_KERNEL_HOST = None
_DECODE_KERNEL_CHECKED = False

#: trtllm-gen context scratch, ONE buffer for the whole process.
#:
#: It is per-launch scratch and layers run sequentially on one stream, so a
#: per-layer buffer bought nothing and cost 128 MB x num_layers (6 GB at 48
#: layers). Worse, it was allocated lazily on the first context prefill --
#: i.e. AFTER vLLM sized the KV cache from free memory -- so that GB never
#: appeared in the budget and came out of the cache's margin at serving time.
#: Allocated eagerly at the first impl construction instead, which is before
#: determine_available_memory() runs.
#:
#: `empty`, not `zeros`: trtllm-gen treats this as opaque scratch and writes
#: before it reads.
_TRTLLM_CTX_WORKSPACE: torch.Tensor | None = None
_TRTLLM_CTX_WORKSPACE_BYTES = 128 * 1024 * 1024


class _DecodeWarmupProvider:
    """CuTeDSL warmup provider for the decode kernel (vLLM first-class hook).

    vLLM's `kernel_warmup` drains `register_cutedsl_warmup_provider` units
    under `enable_cutedsl_warmup` -- the mechanism built for exactly this
    case (JIT'd CuTeDSL kernels), replacing the old `kernel_warmup`
    monkeypatch, which was silently dead under real serving:
    `gpu_worker.py:53` from-imports `kernel_warmup` by name, so rebinding the
    module attribute after that import changed nothing and the only symptom
    was a slow first token.

    One unit per `_compiled` key reachable for the registered geometry, from
    `host.plan_decode_compile_variants` -- which shares its construction with
    the production entry, so warmup compiles what live requests hit. vLLM
    holds providers in a WeakSet; the module-level `_DECODE_WARMUP_PROVIDER`
    strong ref keeps this one alive.
    """

    def __init__(self):
        self._geometries: dict[tuple[int, int, int], tuple[int, int]] = {}

    def add_geometry(self, num_qo_heads: int, num_kv_heads: int,
                     head_dim: int, max_model_len: int,
                     max_num_seqs: int) -> None:
        self._geometries[(num_qo_heads, num_kv_heads, head_dim)] = (
            max_model_len, max_num_seqs
        )

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import (
            CuTeDSLCompileUnit,
        )

        host = _load_decode_kernel()
        if host is None or not torch.cuda.is_available():
            return []
        device = torch.device("cuda", torch.cuda.current_device())
        units = []
        for (hq, hkv, d), (mml, mns) in self._geometries.items():
            for plan in host.plan_decode_compile_variants(
                hq, hkv, d, mml, mns, device
            ):
                units.append(CuTeDSLCompileUnit(
                    name=f"resattn_decode[{plan['label']}]",
                    key=("resattn_decode", plan["args"],
                         tuple(sorted(plan["kwargs"].items()))),
                    compile=functools.partial(
                        host.compile_decode_variant, plan
                    ),
                ))
        return units


_DECODE_WARMUP_PROVIDER: _DecodeWarmupProvider | None = None


def _register_decode_warmup(num_qo_heads: int, num_kv_heads: int,
                            head_dim: int) -> None:
    """Register (or extend) the decode-kernel warmup provider. Best effort."""
    global _DECODE_WARMUP_PROVIDER
    try:
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )
    except ImportError:  # pragma: no cover - vLLM layout drift
        return
    cfg = get_current_vllm_config_or_none()
    max_model_len = (
        cfg.model_config.max_model_len
        if cfg is not None and cfg.model_config is not None else 65536
    )
    max_num_seqs = (
        cfg.scheduler_config.max_num_seqs
        if cfg is not None and cfg.scheduler_config is not None else 64
    )
    if _DECODE_WARMUP_PROVIDER is None:
        _DECODE_WARMUP_PROVIDER = _DecodeWarmupProvider()
    _DECODE_WARMUP_PROVIDER.add_geometry(
        num_qo_heads, num_kv_heads, head_dim, max_model_len, max_num_seqs
    )
    register_cutedsl_warmup_provider(_DECODE_WARMUP_PROVIDER)


def _trtllm_ctx_workspace(device: torch.device) -> torch.Tensor:
    """The process-wide trtllm-gen context workspace, allocating on demand."""
    global _TRTLLM_CTX_WORKSPACE
    ws = _TRTLLM_CTX_WORKSPACE
    if ws is None or ws.device != device:
        ws = torch.empty(
            _TRTLLM_CTX_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        _TRTLLM_CTX_WORKSPACE = ws
    return ws


#: prefill 各路径的调用计数。优化 prefill 前先看这个 —— 曾经在没确认
#: 快路径是否被执行的情况下去优化它,结果毫无收益。纯 python 计数,无同步。
PREFILL_PATH_COUNTS = {
    "varlen_flash": 0,
    "trtllm_gen_context": 0,
    "per_request_bf16": 0,
    "per_request_fp4": 0,
}


def split_prefill_requests(ctx_lens, num_decodes: int):
    '把 prefill 请求按"是否已有前缀"分成两组,返回 (fresh, ctxed) 全局下标。'
    fresh, ctxed = [], []
    for j, c in enumerate(ctx_lens):
        (fresh if c == 0 else ctxed).append(num_decodes + j)
    return fresh, ctxed


def context_kernel_eligible(
    q_len: int,
    seq_len: int,
    head_size: int,
    page_size: int = _MMA_PAGE_SIZE,
) -> bool:
    """Whether one prefixed request fits the context-kernel routing box.

    KV tails use the kernel's explicit length mask. Keep this decision pure so
    CPU tests can pin the CUDA-only routing contract.
    """
    ctx_len = seq_len - q_len
    return (
        head_size == 128
        and page_size == _MMA_PAGE_SIZE
        and ctx_len > 0
        and q_len > 0
    )


def context_kernel_enabled() -> bool:
    """Paged context prefill is production-default; false values opt out.

    Opting out routes prefixed requests to the faithful gather_dequant+SDPA
    oracle, which is the reference this path is validated against.
    """
    return os.environ.get("RESIDUE_ATTN_CONTEXT_KERNEL", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


_TRTLLM_CONTEXT = None
_TRTLLM_CONTEXT_CHECKED = False


def _load_trtllm_context():
    """Return flashinfer's trtllm-gen context prefill entry, or None.

    The kernel reads the ordinary vLLM NVFP4 cache directly, which is
    byte-identical to what ``_trtllm_pool`` views -- so a launch needs no
    repacking. That is the reason this backend ships only that cache profile.
    """
    global _TRTLLM_CONTEXT, _TRTLLM_CONTEXT_CHECKED
    if _TRTLLM_CONTEXT_CHECKED:
        return _TRTLLM_CONTEXT
    _TRTLLM_CONTEXT_CHECKED = True
    try:
        from flashinfer.prefill import (  # type: ignore[import-not-found]
            trtllm_batch_context_with_kv_cache,
        )
    except Exception:  # pragma: no cover - optional dependency
        return None
    _TRTLLM_CONTEXT = trtllm_batch_context_with_kv_cache
    return _TRTLLM_CONTEXT


def _dense_block_tables(kv_indices, kv_indptr_list, first, last, page_size,
                        seq_lens_list):
    """flashinfer flat page table -> trtllm-gen's dense ``[B, max_pages]``.

    trtllm-gen takes a rectangular block table; vLLM hands out a ragged one
    (flat ``kv_indices`` + per-request ``kv_indptr``). Rows are right-padded
    with 0, which the kernel never reads because ``seq_lens`` bounds it.

    ``kv_indptr_list`` is the HOST copy on purpose: the offsets are only used
    to slice, and reading them from the device tensor cost one sync per
    request inside this loop. The slice itself stays device-to-device.
    """
    n = last - first + 1
    pages = [(s + page_size - 1) // page_size for s in seq_lens_list]
    bt = torch.zeros(
        (n, max(pages)), dtype=torch.int32, device=kv_indices.device
    )
    for r in range(n):
        lo = kv_indptr_list[first + r]
        bt[r, : pages[r]] = kv_indices[lo : lo + pages[r]]
    return bt


def _load_decode_kernel():
    """Return the promoted `nvfp4.cute_attn.host` module, or None (guarded)."""
    global _DECODE_KERNEL_HOST, _DECODE_KERNEL_CHECKED
    if _DECODE_KERNEL_CHECKED:
        return _DECODE_KERNEL_HOST
    _DECODE_KERNEL_CHECKED = True
    try:
        from nvfp4.cute_attn import host  # type: ignore[attr-defined]

        _DECODE_KERNEL_HOST = host
    except Exception as exc:
        _DECODE_KERNEL_HOST = None
        print(
            "[residue] ResAttn: decode kernel `nvfp4.cute_attn.host` failed to "
            f"import ({type(exc).__name__}: {exc}) -- backend will refuse at "
            "the first decode forward.",
            flush=True,
        )
    return _DECODE_KERNEL_HOST


# --------------------------------------------------------------------------- #
# Byte-pool reinterpretation (wiring doc step 1 "池所有权说明")
# --------------------------------------------------------------------------- #
def _trtllm_pool(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    *,
    k_global_scale: float = 1.0,
    v_global_scale: float = 1.0,
):
    """Zero-copy views over vLLM's production NVFP4/TRT-LLM cache profile."""
    if not _HAS_CUTE_ATTN:
        raise RuntimeError(
            "[residue] ResAttn: nvfp4.cute_attn management layer unavailable "
            f"({_CUTE_ATTN_IMPORT_ERROR!r})."
        )
    ps = _MMA_PAGE_SIZE
    kv = kv_cache.permute(*ResAttnBackend.get_kv_cache_stride_order())
    # Match FlashInfer's production adapter: a singleton H dimension may carry
    # an arbitrary PyTorch stride, while TMA requires canonical aligned strides.
    kv = canonicalize_singleton_dim_strides(kv)
    (k_data, v_data), (k_sf, v_sf) = nvfp4_kv_cache_split_views(kv)
    expected_data = (kv.shape[0], num_kv_heads, ps, head_size // 2)
    expected_sf = (kv.shape[0], num_kv_heads, ps, head_size // _SF_VEC)
    if tuple(k_data.shape) != expected_data or tuple(v_data.shape) != expected_data:
        raise ValueError(
            "[residue] unexpected vLLM NVFP4 data views: "
            f"K={tuple(k_data.shape)} V={tuple(v_data.shape)}, "
            f"expected {expected_data}"
        )
    if tuple(k_sf.shape) != expected_sf or tuple(v_sf.shape) != expected_sf:
        raise ValueError(
            "[residue] unexpected vLLM NVFP4 scale views: "
            f"K={tuple(k_sf.shape)} V={tuple(v_sf.shape)}, expected {expected_sf}"
        )
    return NvFp4PagedKVPool(
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        k_data=k_data,
        v_data=v_data,
        k_sf=k_sf,
        v_sf=v_sf,
        # Conventional D-grouped V needs no cross-token bf16 staging.
        v_staging=torch.empty(0, dtype=torch.bfloat16, device=kv_cache.device),
        page_size=ps,
        layout_profile="trtllm",
        k_global_scale=float(k_global_scale),
        v_global_scale=float(v_global_scale),
    )


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class ResAttnBackend(AttentionBackend):
    """CuTeDSL NVFP4 attention with native and standard cache profiles."""

    accept_output_buffer: bool = True

    
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = ["nvfp4"]

    # KV write happens INSIDE forward() so the prefill bridge can observe an
    # already-populated pool in the same call. The split
    # `do_kv_cache_update(..., slot_mapping)` hook would work now that vLLM
    # owns the write; adopting it is a follow-up, not a correctness issue.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        # 必须与注册所用的枚举成员名一致:vLLM 会拿这个字符串回查
        # AttentionBackendEnum[name](registry.py 的 __getitem__),返回
        # "RES_ATTN" 会得到 "Unknown attention backend"。我们占的是 CUSTOM 槽。
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["ResAttnImpl"]:
        return ResAttnImpl

    @staticmethod
    def get_builder_cls() -> type["ResAttnMetadataBuilder"]:
        return ResAttnMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Design doc D1 (frozen): page_size=128 -- the SF atom layout makes a
        # 128-row M block contiguous in atom order (1024B/head/page, one TMA),
        # and one K MMA tile == one page. page=64 fallback (cp.async SF
        # assembly) is deliberately NOT implemented.
        return [_MMA_PAGE_SIZE]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Same allocation shape as vLLM's existing nvfp4 path (FlashInfer/
        # trtllm-gen): per-token bytes are identical (D/2 data + D/16 SF), so
        # capacity planning / block manager / prefix caching reuse is free.
        # _trtllm_pool() views these bytes zero-copy; nothing reinterprets them.
        #
        # This is a LAYOUT QUERY, so it must not raise. vLLM legitimately
        # calls it with a derived "auto" dtype for layers whose kv_quant_mode
        # is NONE (v1/worker/gpu/attn_utils.py) -- raising here turned a
        # capacity computation into a startup crash. Enforcement lives in the
        # capability declarations vLLM checks at backend selection
        # (supported_kv_cache_dtypes, get_supported_kernel_block_sizes);
        # block_size is vLLM's to derive from the latter.
        full_dim = (
            nvfp4_kv_cache_full_dim(head_size)
            if cache_dtype_str == "nvfp4"
            else head_size
        )
        return (num_blocks, 2, block_size, num_kv_heads, full_dim)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # HND is REQUIRED: per-(page, side, head) byte chunks must be
        # contiguous for the pool reinterpretation and per-page TMA loads.
        if include_num_layers_dimension:
            # The documented contract (v1/attention/backend.py) is to raise
            # when a layered layout is unsupported; returning the 5-D
            # permutation again would let vLLM build a cross-layer pool this
            # backend cannot view.
            raise NotImplementedError(
                "[residue] ResAttn allocates per-layer; no layered layout."
            )
        return (0, 1, 3, 2, 4)

    @classmethod
    def get_required_kv_cache_layout(cls):
        return "HND"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        
        
        
        return [128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        # No per-layer k_scale/v_scale: SF = amax(group)/6 -> e4m3 per
        # 16-group, no global amax (design doc D4, frozen).
        return False

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        
        return capability.major == 10


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass
class ResAttnMetadata(AttentionMetadata):
    """Per-batch metadata for ResAttn (flashinfer-format page table)."""

    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int

    seq_lens: torch.Tensor  # [num_reqs] int32 -- per-CTA varlen (M0)
    block_table: torch.Tensor  # [num_reqs, max_blocks]
    slot_mapping: torch.Tensor  # [num_tokens] (kept for debug/cross-check)

    # flashinfer-format page table (design doc D5 / append API contract)
    kv_indices: torch.Tensor  # [total_pages] int32 flat physical page ids
    kv_indptr: torch.Tensor  # [num_reqs+1] int32

    # Per-request query token offsets in the reordered token stream
    # (decodes first). Prefill 桥(v0)按此切各请求的 chunk。
    query_start_loc: torch.Tensor  # [num_reqs+1] int32

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    query_start_loc_list: list[int]  # [num_reqs+1]
    seq_lens_list: list[int]  # [num_reqs]
    kv_indptr_list: list[int]  # [num_reqs+1]

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # 是否有 prefill 请求带已在池中的前缀(chunked prefill / 前缀复用)。
    # 每步算一次,免得 prefill 桥每层都要同步一次 GPU 去判断走哪条路。
    has_prefill_context: bool


class ResAttnMetadataBuilder(AttentionMetadataBuilder[ResAttnMetadata]):
    """Builds ResAttnMetadata from CommonAttentionMetadata."""

    # TODO(M4 hardening): decode-only CUDA graphs (UNIFORM_SINGLE_TOKEN_DECODE)
    # once the kernel wrapper is promoted and the page-table buffers move to
    # persistent pinned buffers (FlashInferMetadataBuilder pattern). NEVER for
    # the skeleton -- correctness first.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        
        
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

        # DCP would be SILENTLY WRONG, which is worse than unsupported: the
        # builder pages the full seq_lens and ignores dcp_local_seq_lens, so
        # under --decode-context-parallel-size > 1 every rank would read KV it
        # does not own and produce plausible-looking garbage. Refuse at init.
        dcp = getattr(
            vllm_config.parallel_config, "decode_context_parallel_size", 1
        )
        if dcp and dcp > 1:
            raise NotImplementedError(
                "[residue] ResAttn does not support decode context "
                f"parallelism (dcp_size={dcp}): the page-table builder "
                "ignores dcp_local_seq_lens, so each rank would attend over "
                "KV it does not own."
            )


    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ResAttnMetadata:
        cam = common_attn_metadata
        num_reqs = cam.num_reqs

        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cam, decode_threshold=self.reorder_batch_threshold
            )
        )

        ps = _MMA_PAGE_SIZE
        seq_lens = cam.seq_lens[:num_reqs]
        block_table = cam.block_table_tensor[:num_reqs]

        # flashinfer-format page table from vLLM's block table (row-major
        # masked-select keeps per-sequence page order).
        # TODO(M4 hardening): O(num_reqs * max_blocks) mask each step; move to
        # persistent pinned buffers + np cumsum like FlashInferMetadataBuilder
        # (_compute paged_kv buffers) before any perf measurement.
        num_pages = (seq_lens + (ps - 1)) // ps  # [num_reqs]
        max_blocks = block_table.shape[1]
        page_mask = (
            torch.arange(max_blocks, device=block_table.device).unsqueeze(0)
            < num_pages.unsqueeze(1)
        )
        kv_indices = block_table[page_mask].to(torch.int32)
        kv_indptr = torch.zeros(
            num_reqs + 1, dtype=torch.int32, device=block_table.device
        )
        kv_indptr[1:] = num_pages.cumsum(0, dtype=torch.int32)

        # The same quantities on the host, derived from vLLM's own CPU copies
        # so nothing is read back off the device. The page counts are recomputed
        # here rather than copied from `num_pages` for that reason -- it is host
        # arithmetic on host data, not a transfer.
        seq_lens_list = cam.seq_lens_cpu[:num_reqs].tolist()
        qsl_list = cam.query_start_loc_cpu[: num_reqs + 1].tolist()
        kv_indptr_list = [0]
        for _s in seq_lens_list:
            kv_indptr_list.append(kv_indptr_list[-1] + (_s + ps - 1) // ps)

        return ResAttnMetadata(
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=cam.slot_mapping,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            query_start_loc=cam.query_start_loc[: num_reqs + 1],
            query_start_loc_list=qsl_list,
            seq_lens_list=seq_lens_list,
            kv_indptr_list=kv_indptr_list,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            # Host arithmetic on the host copies. The previous form built this
            # from device tensors and forced a sync via bool(...any()).
            has_prefill_context=any(
                seq_lens_list[i] > qsl_list[i + 1] - qsl_list[i]
                for i in range(num_decodes, num_reqs)
            ),
        )


# --------------------------------------------------------------------------- #
# Impl
# --------------------------------------------------------------------------- #
class ResAttnImpl(AttentionImpl[ResAttnMetadata]):
    "Decode/context attention on vLLM's paged NVFP4 cache."

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        # vLLM forwards per-model extra_impl_args (use_alibi_sqrt, block_m,
        # ...) through **kwargs. Swallowing them means a model that NEEDS one
        # serves with it silently ignored -- reject by name instead, exactly
        # like the `unsupported` dict below does for its set.
        if kwargs:
            raise NotImplementedError(
                f"[residue] ResAttn does not implement extra impl args: "
                f"{sorted(kwargs)}."
            )
        # kv_cache_dtype is enforced by the capability declaration
        # (supported_kv_cache_dtypes = ["nvfp4"] on the backend class): vLLM
        # checks it at backend selection, before any impl is built. A second
        # raise here only duplicated that with a different error message.
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "[residue] ResAttn supports decoder self-attention only."
            )
        vllm_config = get_current_vllm_config_or_none()

        # 滑窗:kernel 不实现窗口掩码,但当窗口 ≥ max_model_len 时窗口永远
        # 截不到任何 token,忽略它与实现它逐位等价 —— 这才是可以放宽的条件,
        # 而不是"反正提示词短"。窗口真的会生效时仍然拒绝。
        if sliding_window is not None:
            max_len = (
                vllm_config.model_config.max_model_len
                if vllm_config is not None
                else None
            )
            win = (
                sliding_window[0]
                if isinstance(sliding_window, (tuple, list))
                else sliding_window
            )
            if max_len is not None and win is not None and win >= max_len:
                print(
                    f"[residue] ResAttn: sliding_window={win} >= max_model_len="
                    f"{max_len}; the window can never truncate, ignoring it.",
                    flush=True,
                )
                sliding_window = None
            else:
                raise NotImplementedError(
                    f"[residue] ResAttn does not implement sliding-window "
                    f"masking (window={win}, max_model_len={max_len}). Serve "
                    f"with --max-model-len <= {win} to make the window a no-op, "
                    f"or use a model without sliding-window attention."
                )

        unsupported = {
            "alibi_slopes": alibi_slopes,
            "logits_soft_cap": logits_soft_cap,
            "sinks": sinks,
            "kv_sharing_target_layer_name": kv_sharing_target_layer_name,
        }
        offending = [k for k, v in unsupported.items() if v is not None]
        if offending:
            raise NotImplementedError(
                f"[residue] ResAttn does not support: {offending}."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # D4: the pool views never change after allocation -- kv_cache is
        # allocated once per layer and the K/V global scales are fixed after
        # weight load -- but _trtllm_pool was being rebuilt on EVERY forward
        # of EVERY layer: a permute + canonicalize + four as_strided + one
        # empty alloc per layer per step, over bytes that never move. Cache
        # keyed on (data_ptr, scales) so a fresh cache or changed scale still
        # rebuilds.
        self._pool: NvFp4PagedKVPool | None = None
        self._pool_key: tuple | None = None

        
        
        
        self._decode_reduction_workspace: tuple[torch.Tensor, ...] | None = None

        # Claim the context workspace NOW, while impls are being constructed --
        # that is before determine_available_memory() sizes the KV cache, so
        # these 128 MB are inside vLLM's budget instead of being taken out of
        # the cache's margin at the first long prefill. Idempotent and shared:
        # every layer gets the same buffer.
        if torch.cuda.is_available():
            _trtllm_ctx_workspace(torch.device("cuda", torch.cuda.current_device()))

        # Decode-kernel warmup rides vLLM's own CuTeDSL warmup hook; see
        # _DecodeWarmupProvider for why the old monkeypatch could not work.
        _register_decode_warmup(
            self.num_heads, self.num_kv_heads, self.head_size
        )

    # -------------------------------------------------------------- helpers
    def _get_decode_reduction_workspace(
        self,
        host,
        splits: int,
        batch: int,
        device: torch.device,
        reduction: str = "kernel",
    ):
        # The CuTe kernel pads each GQA group to its supported N tile. Mirror
        # that effective head count here; using the model's unpadded HQ would
        # make the persistent workspace fail host validation for GQA < 16.
        effective_heads = self.num_kv_heads * host._padded_group_size(
            self.num_heads, self.num_kv_heads
        )
        wanted = [
            (splits, batch, 1, effective_heads, self.head_size),
            (splits, batch, 1, effective_heads),
            (splits, batch, 1, effective_heads),
        ]
        if reduction == "gmem_fused":
            wanted.append((batch * effective_heads,))
        elif reduction != "kernel":
            raise ValueError(f"unsupported persistent reduction: {reduction}")
        workspace = self._decode_reduction_workspace
        if (
            workspace is None
            or len(workspace) != len(wanted)
            or any(tuple(t.shape) != shape for t, shape in zip(workspace, wanted))
            or any(t.device != device for t in workspace)
        ):
            allocator = (
                host.allocate_qmul4_gmem_reduction_workspace
                if reduction == "gmem_fused"
                else host.allocate_qmul4_reduction_workspace
            )
            workspace = allocator(
                splits, batch, effective_heads, self.head_size, device
            )
            self._decode_reduction_workspace = workspace
        return workspace

    # -------------------------------------------------------------- forward
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ResAttnMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        
        
        
        
        
        
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "[residue] ResAttn emits BF16; fused NVFP4 attention output "
                "was retired. Declare fused_output_quant_supported() and let "
                "vLLM's AttnQuantFusionPass rewrite the graph instead."
            )
        if output is None:
            output = torch.zeros(
                query.shape[0],
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            # Profiling / dummy run, before the KV cache exists.
            return output.fill_(0)

        md = attn_metadata
        num_all = md.num_actual_tokens
        num_dec = md.num_decode_tokens
        if num_all == 0:
            return output.fill_(0)
        q_all = query[:num_all].view(num_all, self.num_heads, self.head_size)
        k_all = key[:num_all].view(num_all, self.num_kv_heads, self.head_size)
        v_all = value[:num_all].view(num_all, self.num_kv_heads, self.head_size)

        out3 = (
            output
            if output.ndim == 3
            else output.view(-1, self.num_heads, self.head_size)
        )

        # vLLM owns the write. Its reshape_and_cache_flash quantizes and scatters
        # the new tokens; _trtllm_pool is then a zero-copy view of that result.
        self._append_trtllm_kv(
            layer, k_all, v_all, kv_cache, md.slot_mapping[:num_all]
        )
        k_scale = getattr(layer, "_k_scale_float", 1.0)
        v_scale = getattr(layer, "_v_scale_float", 1.0)
        pool_key = (kv_cache.data_ptr(), k_scale, v_scale)
        if self._pool_key != pool_key:
            self._pool = _trtllm_pool(
                kv_cache,
                self.num_kv_heads,
                self.head_size,
                k_global_scale=k_scale,
                v_global_scale=v_scale,
            )
            self._pool_key = pool_key
        pool = self._pool
        # Calibrated Q scale, the same value vLLM's own fp8 attention uses
        # (flashinfer.py: `bmm1_scale *= layer._q_scale_float`). 1.0 means the
        # checkpoint carries no calibration, in which case the context path
        # derives one on-device instead -- see _attend_context_requests.
        q_scale = float(getattr(layer, "_q_scale_float", 1.0) or 1.0)

        if num_dec > 0:
            self._decode_forward(
                q_all,
                k_all,
                v_all,
                pool,
                md,
                out3,
            )
        if md.num_prefill_tokens > 0:
            self._prefill_bridge(q_all, k_all, v_all, pool, md, out3,
                                 q_scale=q_scale)
        return output

    # ------------------------------------------------------------ decode path
    def _decode_forward(
        self,
        q_all,
        k_all,
        v_all,
        pool,
        md,
        out3,
    ) -> None:
        """One new token per request: append it, then run the paged fp4 kernel.

        Emits BF16. The packed-NVFP4 variant was retired with the fused-output
        bridge (see forward()); O-proj quantizes for itself.
        """
        n = md.num_decode_tokens
        # Avoid manufacturing a distinct full-range view in the decode-only
        # case.  Besides preserving the writer's exact input object, this
        # keeps the common one-token-per-request path free of needless view
        # metadata; mixed decode/prefill batches still take the prefix slice.
        q_dec = q_all if q_all.shape[0] == n else q_all[:n]
        k_dec = k_all if k_all.shape[0] == n else k_all[:n]
        v_dec = v_all if v_all.shape[0] == n else v_all[:n]

        # The new token(s) are already in the pool: forward() ran vLLM's
        # reshape_and_cache_flash before dispatching here.
        host = _load_decode_kernel()
        if host is None:
            raise RuntimeError(
                "[residue] ResAttn decode kernel not available: "
                "`nvfp4.cute_attn.host` could not be imported. Check that "
                "csrc/nvfp4 is on PYTHONPATH and nvidia-cutlass-dsl is "
                "installed (Blackwell GPU required)."
            )

        # The kernel is compiled with out_dtype=BFloat16; write straight into
        # the output slice when it already matches, else stage and cast.
        dec_out = out3[:n]
        kern_out = (
            dec_out
            if dec_out.dtype is torch.bfloat16 and dec_out.is_contiguous()
            else torch.empty_like(dec_out, dtype=torch.bfloat16)
        )
        reduction_workspace = None
        
        
        
        if hasattr(host, "plan_qmul4_decode"):
            splits, reduction = host.plan_qmul4_decode(
                n,
                self.num_kv_heads,
                md.max_seq_len,
                q_dec.device,
            )
            if reduction == "kernel":
                reduction_workspace = self._get_decode_reduction_workspace(
                    host, splits, n, q_dec.device
                )
        host.run_paged_fp4_decode_qmul4_tmem(
            q_dec,
            pool,
            pool.v_data,
            pool.v_sf,
            md.kv_indices,
            md.kv_indptr,
            md.seq_lens[: md.num_decodes],
            self.scale * pool.k_global_scale,
            out=kern_out,
            max_seq_len=md.max_seq_len,
            k_sf_mode="linear",
            value_scale=pool.v_global_scale,
            reduction_workspace=reduction_workspace,
        )
        if kern_out is not dec_out:
            dec_out.copy_(kern_out)

    # ----------------------------------------------------------- prefill path
    def _prefill_bridge(self, q_all, k_all, v_all, pool, md, out3,
                        q_scale: float = 1.0) -> None:
        """Prefill: append the chunk, then route by the source of readable KV.

        Fresh requests keep compute in live-bf16 varlen FlashAttention. A
        request with an existing prefix has no live activation for that prefix;
        when enabled it reads the paged pool directly with the native FP4
        context kernel, otherwise it uses gather_dequant+SDPA as an oracle.

        Two sources for the K/V that attention reads, picked by
        ``RESIDUE_ATTN_PREFILL_EXACT``:

        * default (fast): the incoming bf16 K/V. Fresh prefill attends to the
          tokens it just computed, so this differs from the cache only by
          quantization noise, and it keeps the flash SDPA fast path.
        * ``=1`` (faithful): read the fp4 cache back and attend on that, i.e.
          exactly the bytes decode will see. Slower (a gather per request,
          and a materialized mask) but it makes a quantization or layout bug
          in the write path visible in the model output. Use it when
          validating, not when measuring.

        Chunked prefill (context already in the pool) always takes the
        faithful path for the prefix -- the bf16 prefix is simply not in hand.
        """
        dev = q_all.device
        qsl = md.query_start_loc
        # Routing is host logic, so read the host copies the builder already
        # materialised. These used to come off the device tensors here and in
        # every callee, costing several syncs per layer per step.
        qsl_l = md.query_start_loc_list
        seq_l = md.seq_lens_list
        n_pref = len(seq_l) - md.num_decodes
        if n_pref <= 0:
            return

        # The prefill chunk is already in the pool (forward() ran vLLM's
        # reshape_and_cache_flash); only the routing decision is left.
        n_reqs = len(seq_l)
        q_lens_l = [
            qsl_l[i + 1] - qsl_l[i] for i in range(md.num_decodes, n_reqs)
        ]
        ctx_lens_l = [
            seq_l[md.num_decodes + j] - q for j, q in enumerate(q_lens_l)
        ]
        t_first = qsl_l[md.num_decodes]

        # --- attention ---
        # 三条路,优先级从上到下:
        #  1) varlen flash(CUDA,fresh prefill):所有请求一次调用,原生吃变长
        #     与 GQA。早先是逐请求 SDPA + repeat_interleave 把 KV 撑成 g 倍,
        #     200 请求 × 36 层 = 7200 次迭代外加几十 GB 无谓访存。
        #  2) 逐请求 SDPA on bf16:CPU 回退(flash 是 CUDA-only),供 CPU 测试用。
        #  3) 逐请求 SDPA on 读回的 fp4:前缀已在池里、bf16 原值不在手上时的
        #     唯一选择(chunked prefill / 前缀复用),也是 EXACT=1 的验证路径。
        exact = os.environ.get("RESIDUE_ATTN_PREFILL_EXACT") == "1"
        use_flash = q_all.is_cuda and not exact

        if use_flash and not md.has_prefill_context:
            # 整批都是 fresh prefill:一次调用搞定,且**不做任何 CPU 同步**。
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            PREFILL_PATH_COUNTS["varlen_flash"] += 1
            cu = (qsl[md.num_decodes : n_reqs + 1] - t_first).to(torch.int32)
            max_q = max(q_lens_l)
            out3[t_first:] = flash_attn_varlen_func(
                q=q_all[t_first:],
                k=k_all[t_first:],
                v=v_all[t_first:],
                max_seqlen_q=max_q,
                cu_seqlens_q=cu,
                max_seqlen_k=max_q,
                cu_seqlens_k=cu,
                softmax_scale=self.scale,
                causal=True,
            ).to(out3.dtype)
            return

        if use_flash and md.has_prefill_context:
            
            
            
            
            
            
            
            
            
            
            
            fresh, ctxed = split_prefill_requests(ctx_lens_l, md.num_decodes)

            if fresh:
                from vllm.vllm_flash_attn import flash_attn_varlen_func

                PREFILL_PATH_COUNTS["varlen_flash"] += 1
                spans = [(qsl_l[i], qsl_l[i + 1]) for i in fresh]
                idx = torch.cat(
                    [torch.arange(a, b, device=dev, dtype=torch.int64) for a, b in spans]
                )
                lens = [b - a for a, b in spans]
                cu = torch.tensor(
                    [0, *itertools.accumulate(lens)],
                    device=dev,
                    dtype=torch.int32,
                )
                max_q = max(lens)
                out3[idx] = flash_attn_varlen_func(
                    q=q_all[idx],
                    k=k_all[idx],
                    v=v_all[idx],
                    max_seqlen_q=max_q,
                    cu_seqlens_q=cu,
                    max_seqlen_k=max_q,
                    cu_seqlens_k=cu,
                    softmax_scale=self.scale,
                    causal=True,
                ).to(out3.dtype)

            if ctxed:
                fallback = ctxed
                
                
                
                
                if context_kernel_enabled():
                    fallback = self._attend_context_requests(
                        q_all, pool, md, out3, only=ctxed, q_scale=q_scale
                    )
                if fallback:
                    PREFILL_PATH_COUNTS["per_request_fp4"] += 1
                    self._attend_per_request(
                        q_all, k_all, v_all, pool, md, out3, only=fallback
                    )
            return

        if not q_all.is_cuda:
            PREFILL_PATH_COUNTS["per_request_bf16"] += 1
            # CPU 回退只在 fresh 语义下有意义(bf16 原值即全部 KV)
            pool_arg = pool if (exact or md.has_prefill_context) else None
            self._attend_per_request(q_all, k_all, v_all, pool_arg, md, out3)
            return

        PREFILL_PATH_COUNTS["per_request_fp4"] += 1
        self._attend_per_request(q_all, k_all, v_all, pool, md, out3)

    def _attend_context_requests(
        self, q_all, pool, md, out3, only, q_scale: float = 1.0
    ) -> list[int]:
        "Run eligible prefixed requests with trtllm-gen's context kernel."
        wanted = set(only)
        if not wanted:
            return list(only)
        trtllm_ctx = _load_trtllm_context()
        if trtllm_ctx is None:
            return list(only)

        # Host copies from the builder -- these three lines used to be three
        # device syncs, taken once per layer per step.
        qsl_l = md.query_start_loc_list
        seq_l = md.seq_lens_list
        indptr_l = md.kv_indptr_list
        fallback: list[int] = []
        eligible: list[int] = []
        for i in range(md.num_decodes, len(seq_l)):
            if i not in wanted:
                continue
            t0, t1 = int(qsl_l[i]), int(qsl_l[i + 1])
            q_len = t1 - t0
            seq_len = int(seq_l[i])
            if not context_kernel_eligible(
                q_len, seq_len, self.head_size, pool.page_size
            ):
                fallback.append(i)
                continue

            lo = int(indptr_l[i])
            logical_pages = (seq_len + pool.page_size - 1) // pool.page_size
            hi = lo + logical_pages
            if hi > int(indptr_l[i + 1]):
                # Metadata inconsistency is safer on the established fallback
                # than as an out-of-bounds page-table read in a TMA kernel.
                fallback.append(i)
                continue
            eligible.append(i)

        # One grouped launch per contiguous run. In the common all-context
        # batch this is exactly one launch; mixed fresh/context batches may
        # form several runs because their page-table offsets are noncontiguous.
        runs: list[list[int]] = []
        for i in eligible:
            if not runs or i != runs[-1][-1] + 1:
                runs.append([i])
            else:
                runs[-1].append(i)
        for run in runs:
            first, last = run[0], run[-1]
            t0r, t1r = int(qsl_l[first]), int(qsl_l[last + 1])
            run_seq = [int(seq_l[i]) for i in run]
            qr = q_all[t0r:t1r]
            # Q scale WITHOUT a host round trip. The old form was
            # `float(<qr>.abs().amax()) / 448.0`, which synced the stream once
            # per run per layer per step, materialised a full-size abs() temp,
            # and read Q twice.
            #
            # Calibrated scale when the checkpoint has one -- that is vLLM's
            # fp8-attention practice and is CUDA-graph clean. Otherwise a
            # single fused device-side reduction whose result stays on the GPU:
            # bmm1_scale accepts a float32 tensor, so nothing has to come back.
            if q_scale != 1.0:
                q_s = q_scale
            else:
                q_s = torch.linalg.vector_norm(
                    qr, ord=float("inf"), dtype=torch.float32
                ).div_(448.0)
                # Preserve the old `... or 1.0`: an all-zero (or denormal) Q
                # must scale by 1, not by ~0. clamp_min to a tiny epsilon
                # instead would divide Q by ~0 and overflow fp8.
                q_s = torch.where(q_s > 0, q_s, torch.ones_like(q_s))
            q8 = (qr / q_s).to(torch.float8_e4m3fn)
            o8 = torch.empty_like(q8)
            cu_q = md.query_start_loc[first : last + 2] - int(qsl_l[first])
            cu_kv = torch.zeros(len(run) + 1, dtype=torch.int32, device=qr.device)
            cu_kv[1:] = torch.tensor(
                run_seq, dtype=torch.int32, device=qr.device
            ).cumsum(0)
            trtllm_ctx(
                q8,
                (pool.k_data, pool.v_data),
                kv_cache_sf=(pool.k_sf, pool.v_sf),
                workspace_buffer=_trtllm_ctx_workspace(qr.device),
                block_tables=_dense_block_tables(
                    md.kv_indices, md.kv_indptr_list, first, last,
                    pool.page_size, run_seq,
                ),
                seq_lens=md.seq_lens[first : last + 1].to(torch.int32),
                max_q_len=max(int(qsl_l[i + 1]) - int(qsl_l[i]) for i in run),
                max_kv_len=max(run_seq),
                batch_size=len(run),
                bmm1_scale=q_s * (self.scale * pool.k_global_scale),
                bmm2_scale=pool.v_global_scale,
                cum_seq_lens_q=cu_q.to(torch.int32),
                cum_seq_lens_kv=cu_kv,
                kv_layout="HND",
                causal=True,
                out=o8,
            )
            out3[t0r:t1r] = o8.to(out3.dtype)
            PREFILL_PATH_COUNTS["trtllm_gen_context"] += len(run)
        return fallback

    def _attend_per_request(
        self, q_all, k_all, v_all, pool, md, out3, only=None
    ) -> None:
        """逐请求 SDPA。``pool`` 非 None 时注意力跑在**读回来的 fp4** 上。

        慢,但它是语义参考:varlen flash 的快路径必须与 pool=None 的这一条
        等价(GPU 测试守着),而 pool 非 None 的那一条是前缀复用场景下唯一
        可行的算法 —— 根治要靠原生 fp4 prefill kernel。

        ``only``:只处理这些请求下标(全局下标,含 decode 偏移)。混合批下
        由 ``_prefill_bridge`` 把 ctx=0 的请求分给 flash、ctx>0 的分到这里。
        ``None`` = 全部 prefill 请求。
        """
        dev = q_all.device
        g = self.num_queries_per_kv
        qsl_l = md.query_start_loc_list
        seq_l = md.seq_lens_list
        wanted = None if only is None else set(only)
        for i in range(md.num_decodes, len(seq_l)):
            if wanted is not None and i not in wanted:
                continue
            t0, t1 = int(qsl_l[i]), int(qsl_l[i + 1])
            q_len = t1 - t0
            if q_len <= 0:
                continue
            seq_len = int(seq_l[i])
            ctx = seq_len - q_len
            if pool is not None:
                k_src, v_src = gather_dequant_trtllm(
                    pool, seq_len, i, md.kv_indices, md.kv_indptr
                )
            else:
                k_src, v_src = k_all[t0:t1], v_all[t0:t1]
                seq_len, ctx = q_len, 0
            kh = k_src.repeat_interleave(g, dim=1).transpose(0, 1).unsqueeze(0)
            vh = v_src.repeat_interleave(g, dim=1).transpose(0, 1).unsqueeze(0)
            qh = q_all[t0:t1].float().transpose(0, 1).unsqueeze(0)
            key_pos = torch.arange(seq_len, device=dev).unsqueeze(0)
            qry_pos = (ctx + torch.arange(q_len, device=dev)).unsqueeze(1)
            o = torch.nn.functional.scaled_dot_product_attention(
                qh.float(), kh.float(), vh.float(),
                attn_mask=(key_pos <= qry_pos).unsqueeze(0).unsqueeze(0),
                scale=self.scale,
            )
            out3[t0:t1] = o[0].transpose(0, 1).to(out3.dtype)
