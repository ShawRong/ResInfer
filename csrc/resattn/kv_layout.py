# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Views of the stock NVFP4 KV cache consumed by ResAttn decode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

SF_VEC = 16  # Values per NVFP4 block scale.
PAGE_SIZE = 128  # Page size supported by the current ResAttn kernels.


@dataclass
class NvFp4PagedKVPool:
    """Borrowed K/V data and scale views; vLLM owns allocation and writes.

    Data shape is [pages, heads, page_size, head_dim/2]; scale shape is
    [pages, heads, page_size, head_dim/16]. Storage strides and TRT-LLM's
    V-scale interleave are retained. Construction performs no tensor work.
    """

    num_kv_heads: int
    head_dim: int
    k_data: torch.Tensor
    v_data: torch.Tensor
    k_sf: torch.Tensor
    v_sf: torch.Tensor
    page_size: int = PAGE_SIZE
    k_global_scale: float = 1.0
    v_global_scale: float = 1.0
    layout_profile: ClassVar[str] = "trtllm"
