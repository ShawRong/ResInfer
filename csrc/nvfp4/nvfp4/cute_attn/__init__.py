# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'ResAttn:MMA-native paged NVFP4 KV cache 管理层 + decode attention 接入。'

from .kv_layout import (  # noqa: F401
    PAGE_SIZE,
    PI16,
    SF_VEC,
    NvFp4PagedKVPool,
    sf_atom_gather,
    sf_atom_offsets,
    sf_atom_scatter,
)
from .append import (  # noqa: F401
    build_v_groups,
    bulk_quantize_paged,
    gather_dequant,
    gather_dequant_trtllm,
)
