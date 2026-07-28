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
 * Both projects are licensed under the Apache License, Version 2.0.
 * This file contains minor adaptations for project-specific use.
 */

#pragma once

#include <torch/all.h>

// Dispatch macro for half-precision types (half and bfloat16)
#define VLLM_DISPATCH_CASE_HALF_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))
