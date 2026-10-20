# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Linear callables shared by latency and real-input precision benchmarks."""

from collections.abc import Callable
from typing import NamedTuple


KEXT_BASELINES = ("fi_nvfp4_cutedsl", "fi_mxfp4_mxfp8_w4a8",
                  "trtllm_nvfp4_fp8_w4a8", "trtllm_mxfp4_mxfp8_w4a8")

W4A8_CONFIG = dict(mma_sm=1, tile_m=128, tile_n=128, tile_k=128, swap_ab=True)
# Other arms use FlashInfer's Python AutoTuner.
STATIC_SELECTION = {
    "fi_mxfp4_mxfp8_w4a8": "official_default",
    "trtllm_nvfp4_fp8_w4a8": "trtllm_internal",
    "resinfer_mxfp4_mext_r1": "fixed",
}
ARM_FORMAT = {
    "mext_r1": ("NVFP4", "NVFP4+residue", "residue Fold"),
    "fi_nvfp4_cutlass": ("NVFP4", "NVFP4", "FlashInfer CUTLASS"),
    "fi_nvfp4_cutedsl": ("NVFP4", "NVFP4", "FlashInfer CuTeDSL"),
    "resinfer_mxfp4_mext_r1": ("MXFP4", "MXFP4+residue", "ResInfer Fold fixed"),
    "resinfer_mxfp4_mext_r1_tuned": ("MXFP4", "MXFP4+residue", "ResInfer Fold tuned"),
    "fi_mxfp4_mxfp4_stock": ("MXFP4", "MXFP4", "FlashInfer CuTeDSL"),
    "fi_mxfp4_mxfp4_cutedsl_quant": ("MXFP4", "MXFP4", "FlashInfer CuTeDSL quant+GEMM"),
    "fi_mxfp4_mxfp8_w4a8": ("MXFP4", "MXFP8", "FlashInfer grouped"),
    "trtllm_nvfp4_fp8_w4a8": ("NVFP4", "FP8", "TensorRT-LLM"),
    "trtllm_mxfp4_mxfp8_w4a8": ("MXFP4", "MXFP8", "TensorRT-LLM"),
}


def make_runners(x, weights, arms):
    """Bind each backend to its weight layout and preallocated output buffers."""
    import torch
    from flashinfer import SfLayout, mm_fp4, mxfp4_quantize, mxfp8_quantize, nvfp4_quantize
    from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise
    from .mext_trtllm_adapter import linear as trtllm_w4a8_linear

    if {"resinfer_mxfp4_mext_r1", "resinfer_mxfp4_mext_r1_tuned"} & set(arms):
        from quant.flashinfer_integration import mxfp4_linear_adapter
    m, k = x.shape
    n, device = weights["reference"].shape[0], x.device
    w_nv, ws_nv, ws_mext, wgs = weights["nvfp4"]
    w_mx, ws_mx, ws_mx_flat, w_mx_batch, ws_mx_batch = weights["mxfp4"]
    mask = torch.empty(1, device=device, dtype=torch.uint8)
    xgs = (448.0 * 6.0 / x.float().abs().max()).reshape(())
    alpha = (1.0 / (xgs * wgs)).reshape(1).float()
    if weights["trtllm"] is not None:
        w_trt, ws_trt, wgs_trt = weights["trtllm"]
        # TRT's static FP8 op takes the dequantization scale
        # (amax / 448); the FP4 weight producer takes its inverse.
        x_scale_trt = (x.float().abs().max() / 448.0).reshape(1).float()
        alpha_trt = (x_scale_trt / wgs_trt).reshape(1).float()
    outputs = {arm: torch.empty(m, n, device=device, dtype=torch.bfloat16)
               for arm in arms if arm in {
                   "fi_nvfp4_cutlass", "fi_nvfp4_cutedsl",
                   "fi_mxfp4_mxfp4_stock", "fi_mxfp4_mxfp4_cutedsl_quant"}}
    if "fi_mxfp4_mxfp8_w4a8" in arms:
        padded_m = max(4, m)
        indptr = torch.tensor([0, padded_m], device=device, dtype=torch.int32)
        mx_output = torch.empty(padded_m, n, device=device, dtype=torch.bfloat16)

    def mext(a, w, ws):
        return torch.ops.residue.nvfp4_linear(
            a, w, xgs, ws, ws, mask, alpha,
            k, 0, 0, n, True, False, True)

    def fi_cutlass(a, w, ws):
        aq, asc = nvfp4_quantize(a, xgs, sfLayout=SfLayout.layout_128x4)
        return mm_fp4(aq, w.T, asc, ws.T, alpha, torch.bfloat16,
                      outputs["fi_nvfp4_cutlass"], backend="cutlass",
                      use_8x4_sf_layout=False)

    def fi_cutedsl(a, w, ws):
        aq, asc = nvfp4_quantize(a, xgs, sfLayout=SfLayout.layout_128x4)
        return mm_fp4(aq, w.T, asc, ws.T, alpha, torch.bfloat16,
                      outputs["fi_nvfp4_cutedsl"], backend="cute-dsl",
                      use_8x4_sf_layout=False)

    def fi_w4a8(a, w, ws):
        if a.shape[0] < 4:
            padded = torch.zeros(4, a.shape[1], device=a.device, dtype=a.dtype)
            padded[:a.shape[0]].copy_(a)
            a = padded
        aq, asf = mxfp8_quantize(a)
        return group_gemm_mxfp8_mxfp4_nt_groupwise(
            aq, w, asf, ws, indptr, out=mx_output,
            out_dtype=torch.bfloat16, **W4A8_CONFIG)[:m]

    def res_mxfp4_mext(a, w, ws):
        return mxfp4_linear_adapter.tuned_mxfp4_fold(
            a, w, ws, pdl_mode="early", tune=False,
            out_dtype=torch.bfloat16)

    def res_mxfp4_mext_tuned(a, w, ws):
        return mxfp4_linear_adapter.tuned_mxfp4_fold(
            a, w, ws, pdl_mode="early",
            out_dtype=torch.bfloat16)

    def fi_mxfp4_stock(a, w, ws):
        aq, asc = mxfp4_quantize(a)
        return mm_fp4(
            aq, w.T, asc, ws.T, None, torch.bfloat16,
            outputs["fi_mxfp4_mxfp4_stock"], block_size=32,
            backend="cute-dsl", use_nvfp4=False)

    def fi_mxfp4_cutedsl_quant(a, w, ws):
        aq, asc = mxfp4_quantize(a, backend="cute-dsl")
        return mm_fp4(
            aq, w.T, asc, ws.T, None, torch.bfloat16,
            outputs["fi_mxfp4_mxfp4_cutedsl_quant"], block_size=32,
            backend="cute-dsl", use_nvfp4=False)

    runners = {
        "mext_r1": (mext, (x, w_nv, ws_mext)),
        "fi_nvfp4_cutlass": (fi_cutlass, (x, w_nv, ws_nv)),
        "fi_nvfp4_cutedsl": (fi_cutedsl, (x, w_nv, ws_nv)),
        "resinfer_mxfp4_mext_r1": (res_mxfp4_mext, (x, w_mx, ws_mx_flat)),
        "resinfer_mxfp4_mext_r1_tuned": (
            res_mxfp4_mext_tuned, (x, w_mx, ws_mx_flat)),
        "fi_mxfp4_mxfp4_stock": (fi_mxfp4_stock, (x, w_mx, ws_mx)),
        "fi_mxfp4_mxfp4_cutedsl_quant": (
            fi_mxfp4_cutedsl_quant, (x, w_mx, ws_mx)),
        "fi_mxfp4_mxfp8_w4a8": (fi_w4a8, (x, w_mx_batch, ws_mx_batch)),
    }
    if weights["trtllm"] is not None:
        runners["trtllm_nvfp4_fp8_w4a8"] = (
            trtllm_w4a8_linear, (x, w_trt, ws_trt, x_scale_trt, alpha_trt))
    if "trtllm_mxfp4_mxfp8_w4a8" in arms:
        from .mext_trtllm_adapter import make_mxfp8_runner
        runners["trtllm_mxfp4_mxfp8_w4a8"] = (
            make_mxfp8_runner(x.device), (x, w_mx, ws_mx_flat))
    return runners


def selected_tactic(arm):
    """Record Fold selection when available; profiling belongs to FlashInfer."""
    if arm == "mext_r1":
        from quant.flashinfer_integration import sm100_fold_candidates as adapter
    elif arm == "resinfer_mxfp4_mext_r1_tuned":
        from quant.flashinfer_integration import mxfp4_linear_adapter as adapter
    else:
        return {}
    return {"tactic_source": adapter.last_tactic_source(),
            "tactic": adapter.last_tactic_tile()}


class BenchmarkArm(NamedTuple):
    """One callable and its explicit rotating inputs, plus result labels."""
    family: str
    ratio: float
    forward: Callable
    inputs: tuple
    route: str
    policy: object = None
    arm: str = ""

    @property
    def tuning_buckets(self):
        m = self.inputs[0].shape[0]
        # The ratio1 row-pair GEMM has twice the outer Linear's rows.
        return (m, 2 * m) if self.route == "ratio1_prefill" else (m,)

def residue_arm(x, weight, scale, base_scale, scale_inv, alpha, mask,
                *, k, n, num_salient, ratio1, fold_eligible=True):
    """Use the production route, exposing its active weight/scales to rotation."""
    import torch
    from quant.vllm_integration.residue_linear_op import _linear_policy
    from quant.flashinfer_integration.nvfp4_linear_adapter import _route

    policy = _linear_policy(k, num_salient, 0, n, fold_eligible, False, ratio1)
    route = _route(x.shape[0], policy)
    if route in ('sm10x_fold', 'sm120_fold', 'ratio1_prefill'):
        weight, scale = weight[:, :k // 2], base_scale

    def forward(a, w, sf):
        return torch.ops.residue.nvfp4_linear(
            a, w, scale_inv, sf, sf, mask, alpha,
            k, num_salient, 0, n, fold_eligible, False, ratio1)

    return forward, (x, weight, scale), route, policy


def selected_fold_tactic(route, policy):
    """Read the selected Fold configuration; never infer it from log messages."""
    if route == 'sm10x_fold':
        from quant.flashinfer_integration import sm100_fold_candidates as candidates
        enabled = policy.sm10x_tuner
    elif route == 'sm120_fold':
        from quant.flashinfer_integration import sm120_fold_candidates as candidates
        enabled = policy.sm120_tuner
    else:
        return {}
    return {'fold_tactic_source': candidates.last_tactic_source() if enabled else 'offline_table',
            'fold_tactic_tile': candidates.last_tactic_tile()}


def make_kext_arms(x, weights, extended, ratios):
    """Shared prefill comparison: explicit selective KEXT, ratio1 auto and W4A8."""
    import torch
    from .inputs import global_scale

    k, n = x.shape[1], weights["reference"].shape[0]
    xgs = global_scale(x)
    baselines = make_runners(x, weights, KEXT_BASELINES)
    arms = [BenchmarkArm("stock" if name == KEXT_BASELINES[0] else "w4a8",
                        0., *baselines[name], "baseline", arm=name)
            for name in KEXT_BASELINES]
    for ratio in ratios:
        if ratio == 1.0:
            w, _, sf, gs = weights["nvfp4"]
            base_sf, mask, salient = sf, torch.empty(1, device=x.device, dtype=torch.uint8), 0
        else:
            w, sf, base_sf, gs, mask, salient = extended[ratio]
        fn, inputs, route, policy = residue_arm(
            x, w, sf, base_sf, xgs, (1 / (xgs * gs)).reshape(()), mask,
            k=k, n=n, num_salient=salient, ratio1=ratio == 1.0,
            fold_eligible=ratio == 1.0)
        arms.append(BenchmarkArm("kext", ratio, fn, inputs, route, policy,
                                 f"kext_r{ratio:g}"))
    return arms
