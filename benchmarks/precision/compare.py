"""Paired attention and linear comparisons on captured decode inputs."""

import hashlib
import json
import time


LINEAR_ARMS = (
    "fi_nvfp4_cutedsl",
    "mext_r1",
    "fi_mxfp4_mxfp4_stock",
    "resinfer_mxfp4_mext_r1_tuned",
    "fi_mxfp4_mxfp8_w4a8",
    "trtllm_nvfp4_fp8_w4a8",
    "trtllm_mxfp4_mxfp8_w4a8",
)
LINEAR_SCOPE = (
    "same BF16 weights/activations; per-format weight quantization; "
    "total operator error, not calibrated checkpoint recipes"
)


def check_linear_coverage(rows, protocol, weights):
    expected = {
        (case["index"], step, name, arm)
        for case in protocol["cases"] for step in protocol["steps"]
        for name in weights for arm in LINEAR_ARMS
    }
    actual = [(r["index"], r["step"], r["layer"], r["arm"])
              for r in rows if r["kind"] == "mext"]
    observed = set(actual)
    if observed != expected or len(actual) != len(observed):
        raise ValueError(
            f"Linear coverage: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}, "
            f"duplicates={len(actual) - len(observed)}")


def summarize(rows):
    """Pool squared error and reference energy, not per-point NMSE ratios."""
    from collections import defaultdict

    groups = defaultdict(list)
    for row in rows:
        references = ("fp32_reference",) if row["kind"] in ("mext", "kext") else (
            "quantized_kv_reference", "raw_kv_reference")
        for reference in references:
            groups[row["kind"], row["arm"], reference].append(row[reference])
    summary = []
    for (kind, arm, reference), points in sorted(groups.items()):
        sse = sum(p["sse"] for p in points)
        energy = sum(p["reference_energy"] for p in points)
        n = sum(p["n"] for p in points)
        summary.append(dict(kind=kind, arm=arm, reference=reference,
                            points=len(points), n=n, sse=sse, reference_energy=energy,
                            mse=sse / n, nmse=sse / energy if energy else None))
    return summary


def digest(t):
    import torch

    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def compare_attention(
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    md,
    output,
    history,
    case,
    generated,
    name,
    root,
    save_inputs,
):
    """Compare both implementations on one identical decode state."""
    import torch
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    from .reference import attention, metrics, unpack_kv

    rows = []
    t = time.time()
    length = int(md.decode.seq_lens[0])
    q = query[:1].clone()
    assert length == len(case["prompt_ids"]) + generated
    pool = self._pool_for(kv_cache, layer)
    k, v = unpack_kv(pool, md.decode.block_tables[0], length)
    rawk, rawv = (torch.cat(parts, 0) for parts in history)
    assert rawk.shape[0] == length
    ref = attention(q, k, v, self.scale)
    full = attention(q, rawk, rawv, self.scale)
    stock = torch.empty_like(output)
    FlashInferImpl.forward(self, layer, query, key, value, kv_cache, md, stock)
    base = {
        "kind": "resattn",
        "index": case["index"],
        "layer": name,
        "step": generated,
        "length": length,
        "q_sha256": digest(q),
        "k_sha256": digest(k),
        "v_sha256": digest(v),
        "k_global_scale": pool.k_global_scale,
        "v_global_scale": pool.v_global_scale,
    }
    for arm, y in [("resattn", output[:1]), ("stock", stock[:1])]:
        rows.append(
            dict(
                base,
                arm=arm,
                quantized_kv_reference=metrics(y, ref),
                raw_kv_reference=metrics(y, full),
            )
        )
    if save_inputs and generated == 1:
        ids = md.decode.block_tables[0, : (length + pool.page_size - 1) // pool.page_size].long()
        torch.save(
            {
                "q": q.cpu(),
                "k": k.cpu(),
                "v": v.cpu(),
                "rawk": rawk.cpu(),
                "rawv": rawv.cpu(),
                "packed_k": pool.k_data[ids].cpu(),
                "k_sf": pool.k_sf[ids].cpu(),
                "packed_v": pool.v_data[ids].cpu(),
                "v_sf": pool.v_sf[ids].cpu(),
                "k_scale": pool.k_global_scale,
                "v_scale": pool.v_global_scale,
                "stock": stock[:1].cpu(),
                "resattn": output[:1].cpu(),
                "scale": self.scale,
            },
            root / f"attn-{case['index']}-{name}.pt",
        )
    cost = {
        "stage": "attention_reference",
        "seconds": time.time() - t,
        "index": case["index"],
        "step": generated,
        "layer": name,
    }
    return rows, cost


def compare_linear(
    weights, activations, protocol, root, save_inputs, timing, rows, costs, save
):
    """Quantize each real weight once, then compare stock and MEXT on captured inputs."""
    import torch

    from benchmarks.kernel.inputs import quantize_mext_weights
    from benchmarks.kernel.linear import ARM_FORMAT, STATIC_SELECTION, make_runners, selected_tactic
    from benchmarks.kernel.timing import fi_autotuned, timed_detail

    from .reference import metrics

    snapshot_saved = False
    for name, w in weights.items():
        t = time.time()
        whash = digest(w)
        packed = quantize_mext_weights(w, LINEAR_ARMS)
        for (index, step, label), x in activations.items():
            if label != name:
                continue
            xhash = digest(x)
            reference = x.float() @ w.float().T
            snapshot = (
                {
                    "x": x.cpu(),
                    "w": w.cpu(),
                    "reference": reference.cpu(),
                    "outputs": {},
                }
                if save_inputs and not snapshot_saved
                else None
            )
            runners = make_runners(x, packed, LINEAR_ARMS)
            for arm in LINEAR_ARMS:
                fn, inputs = runners[arm]
                selection = STATIC_SELECTION.get(arm, "flashinfer_autotune")
                weight_format, activation_format, provider = ARM_FORMAT[arm]
                with fi_autotuned(fn, inputs, m_bucket=1,
                                  enabled=selection == "flashinfer_autotune"):
                    output = fn(*inputs).clone()
                    item = {
                        "kind": "mext",
                        "index": index,
                        "step": step,
                        "layer": name,
                        "arm": arm,
                        "m": 1,
                        "n": w.shape[0],
                        "k": w.shape[1],
                        "weight_sha256": whash,
                        "input_sha256": xhash,
                        "weight_format": weight_format,
                        "activation_format": activation_format,
                        "provider": provider,
                        "selection": selection,
                        **selected_tactic(arm),
                        "fp32_reference": metrics(output, reference),
                    }
                    # Only one fixed point per projection is timed; reference work stays outside.
                    if (
                        timing
                        and index == protocol["cases"][0]["index"]
                        and step == protocol["steps"][0]
                    ):
                        item["timing"] = timed_detail(fn, inputs, repeats=2)
                rows.append(item)
                if snapshot is not None:
                    snapshot["outputs"][arm] = output.cpu()
            if snapshot is not None:
                torch.save(snapshot, root / "mext-first-real-input.pt")
                snapshot_saved = True
        costs.append(
            {"stage": "mext_projection", "layer": name, "seconds": time.time() - t}
        )
        save("mext")
        print("LINEAR", name, flush=True)


def compare_prefill(weights, activations, metadata, timing, rows, costs, save):
    """Compare calibrated KEXT ratios and W4A8 on each captured real prefill chunk."""
    from benchmarks.kernel.inputs import calibrated_indices, make_extended_weights, quantize_mext_weights
    from benchmarks.kernel.linear import KEXT_BASELINES, STATIC_SELECTION, make_kext_arms, selected_fold_tactic
    from benchmarks.kernel.timing import fi_autotuned, timed_detail, warm_equally
    from .reference import metrics

    ratios = sorted(metadata) + [1.0]
    for name, w in weights.items():
        started = time.time()
        masks = {ratio: calibrated_indices(data, name, w.shape[1], ratio)
                 for ratio, data in metadata.items()}
        extended = make_extended_weights(w, w.shape[1], sorted(metadata), masks)
        packed = quantize_mext_weights(w, KEXT_BASELINES)
        whash = digest(w)
        for (index, start, label), captured in activations.items():
            if label != name:
                continue
            x = captured.to(w.device)
            reference = x.float() @ w.float().T
            xhash = digest(x)
            for arm in make_kext_arms(x, packed, extended, ratios):
                selection = STATIC_SELECTION.get(arm.arm, "flashinfer_autotune")
                with fi_autotuned(arm.forward, arm.inputs, m_bucket=x.shape[0],
                                  enabled=selection == "flashinfer_autotune"):
                    output = arm.forward(*arm.inputs)
                    item = dict(kind="kext", phase="prefill", index=index,
                        token_start=start, token_count=x.shape[0], layer=name,
                        arm=arm.arm, ratio=arm.ratio, route=arm.route, selection=selection,
                        m=x.shape[0], n=w.shape[0], k=w.shape[1],
                        input_sha256=xhash, weight_sha256=whash,
                        fp32_reference=metrics(output, reference))
                    if arm.ratio in masks:
                        item["mask_sha256"] = hashlib.sha256(
                            json.dumps(masks[arm.ratio]).encode()).hexdigest()
                    if arm.policy:
                        item.update(selected_fold_tactic(arm.route, arm.policy))
                    if timing:
                        warm_equally(arm.forward, arm.inputs, calls=20)
                        item["timing"] = timed_detail(arm.forward, arm.inputs)
                rows.append(item)
        costs.append(dict(stage="kext_projection", layer=name, seconds=time.time()-started))
        save("kext")
        print("PREFILL LINEAR", name, flush=True)
    check_prefill_coverage(rows, activations, ratios)


def check_prefill_coverage(rows, activations, ratios):
    from benchmarks.kernel.linear import KEXT_BASELINES

    arms = set(KEXT_BASELINES) | {f"kext_r{ratio:g}" for ratio in ratios}
    expected = {(index, start, name, arm)
                for index, start, name in activations for arm in arms}
    actual = [(r["index"], r["token_start"], r["layer"], r["arm"])
              for r in rows if r["kind"] == "kext"]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Incomplete or duplicate prefill comparisons")
