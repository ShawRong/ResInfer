# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment kernels. Nothing here is on a production path.

Imported lazily on purpose: `import nvfp4` must not trigger a JIT compile.
"""

__all__ = ["moe_fold"]