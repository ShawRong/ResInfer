"""Preserve excluded GDN consumers when AWQ scales their shared norm.

ModelOpt0.45 discovers only enabled quantized consumers. Qwen3.5's in_proj_a/b
are excluded but consume the same scaled norm output as in_proj_qkv/z.
"""
import torch


def capture_consumers(model):
    snapshots = []
    for name, block in model.named_modules():
        attn = getattr(block, "linear_attn", None)
        norm = getattr(block, "input_layernorm", None)
        if attn is None or type(norm).__name__ != "Qwen3_5RMSNorm":
            continue
        consumers = []
        for suffix in ("in_proj_a", "in_proj_b"):
            layer = getattr(attn, suffix, None)
            if layer is None:
                raise ValueError(f"Missing GDN consumer {name}.{suffix}")
            enabled = any(bool(getattr(getattr(layer, q, None), "is_enabled", False))
                          for q in ("input_quantizer", "weight_quantizer"))
            if not enabled:
                consumers.append((name + ".linear_attn." + suffix, layer, layer.weight.detach().clone()))
        if consumers:
            snapshots.append({"name": name + ".input_layernorm", "norm": norm,
                              "before_gamma": norm.weight.detach().float().clone() + 1,
                              "consumers": consumers})
    return snapshots


@torch.no_grad()
def compensate_consumers(snapshots):
    """W' = W * gamma_before/gamma_after; biases are unchanged.

    Use the *stored effective* gamma ratio, accounting for BF16 norm rounding.
    Refuse an irreversible newly-zero channel or an already-modified consumer.
    """
    changed = []
    for snapshot in snapshots:
        before = snapshot["before_gamma"]
        after = snapshot["norm"].weight.detach().float() + 1
        if torch.equal(before, after):
            continue
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError("Non-finite norm gamma")
        if ((after == 0) & (before != 0)).any():
            raise ValueError("Norm fusion erased nonzero channels; cannot compensate")
        multiplier = torch.where(after == 0, torch.ones_like(after), before / after)
        for name, layer, original_weight in snapshot["consumers"]:
            if not torch.equal(layer.weight, original_weight):
                raise ValueError(f"Excluded consumer already changed during fusion: {name}")
            if layer.weight.shape[-1] != multiplier.numel():
                raise ValueError(f"Consumer/norm width mismatch: {name}")
            layer.weight.copy_((original_weight.float() * multiplier).to(layer.weight.dtype))
            changed.append(name)
    return changed
