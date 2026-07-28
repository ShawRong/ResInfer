# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL MExt R1 fold GEMM for SM100-class GPUs (B200/B300).

Requires nvidia-cutlass-dsl and a Blackwell datacenter GPU; import lazily.
"""

from .host import mext_fold_gemm_sm103  # noqa: F401
from .tactics import (  # noqa: F401
    TACTICS,
    default_tactic,
    get_valid_tactics,
    run_fold_tactic,
    warmup_tactics,
)
