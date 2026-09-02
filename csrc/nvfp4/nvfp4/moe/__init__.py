# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Production MoE kernels: the CuTeDSL graft chain.

Promoted out of nvfp4.experiment once the chain became the default
(RESIDUE_MOE_IMPL=cutedsl). Contents:

- residue_glue: input quant + residue interleave for the grouped GEMMs
- blockscaled_contiguous_gather_grouped_gemm_act_fusion (g1) and
  blockscaled_contiguous_grouped_gemm_finalize_fusion (g2): FlashInfer-style
  wrappers over the vendored TRT-LLM kernels in blackwell/
- swiglu_quant: inter-GEMM activation quant
- blackwell/: the vendored CuTeDSL kernel bodies (sm100)

Imported lazily on purpose: `import nvfp4` must not trigger a JIT compile.
"""
