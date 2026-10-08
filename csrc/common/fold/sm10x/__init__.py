# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL MExt R1 fold GEMM for SM100 GPUs (B200).

Requires nvidia-cutlass-dsl and a Blackwell datacenter GPU; import lazily.

`run_fold_default` is the serving entry (heuristic, no measurement);
`run_fold_autotuned` measures the ranked candidates once per shape and caches.
See tactics.py for how the two relate.
"""

from .host import mext_fold_gemm  # noqa: F401
from .tactics import (  # noqa: F401
    autotuned_tactic_for,
    kernel_arch_for_capability,
    run_fold_autotuned,
    run_fold_default,
    warmup_fold_default,
)
