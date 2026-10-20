// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
/*
 * Type dispatch utilities for NVFP4 quantization kernels.
 *
 * Adapted from the vLLM project:
 *   https://github.com/vllm-project/vllm
 *
 * The vLLM project includes code adapted from:
 *   PyTorch ATen/Dispatch.h
 *   https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 *
 * vLLM is licensed under Apache-2.0; PyTorch is licensed under BSD-3-Clause.
 * PyTorch copyright notices and license: ../../licenses/BSD-3-Clause-PyTorch.txt.
 * This file contains minor adaptations for project-specific use.
 */

#pragma once

#include <torch/all.h>


#define VLLM_DISPATCH_CASE_HALF_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))
