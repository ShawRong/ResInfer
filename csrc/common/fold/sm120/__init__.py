# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL NVFP4 MExt-R1 fold GEMM for SM120 (GeForce Blackwell, RTX 5090).

Optional alternative to the C++ CUTLASS SM120 fold op. Requires
nvidia-cutlass-dsl; import lazily (the module JITs on first use).
"""

from .host import compile_fold_gemm_sm120, mext_fold_gemm_sm120  # noqa: F401
from .tactics import (  # noqa: F401
    act_pad_multiple,
    run_fold_default,
    select_tile,
    supports_capability,
    warmup_fold_default,
)
