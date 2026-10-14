"""Paired attention and linear comparisons on captured decode inputs."""

import hashlib
import time


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
    original,
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
    ref64 = attention(q, k, v, self.scale, torch.float64)
    reference_metrics = metrics(ref, ref64)
    nmse = reference_metrics["nmse"]
    assert nmse is not None and nmse < 1e-9
    ids = md.decode.block_tables[0, : (length + 127) // 128].long()
    buffers = [pool.k_data, pool.k_sf, pool.v_data, pool.v_sf]
    before = [x[ids].clone() for x in buffers]
    stock = torch.empty_like(output)
    repeat = torch.empty_like(output)
    FlashInferImpl.forward(self, layer, query, key, value, kv_cache, md, stock)
    FlashInferImpl.forward(self, layer, query, key, value, kv_cache, md, repeat)
    assert torch.equal(stock, repeat)
    assert all(
        torch.equal(x[ids].view(torch.uint8), b.view(torch.uint8))
        for x, b in zip(buffers, before)
    )
    original(self, layer, query, key, value, kv_cache, md, repeat)
    assert torch.equal(output, repeat)
    graph_check = None
    if generated == 1:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._decode_forward(q, pool, md, repeat)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(output[:1], repeat[:1])
        graph_check = True
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
        "repeat_equal": True,
        "cache_unchanged": True,
        "operator_graph_equal": graph_check,
        "fp32_vs_fp64": reference_metrics,
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

    # Isolated activation-residue ablation on real W/X, not a checkpoint-recipe comparison.
    from benchmarks.kernel.inputs import quantize_mext_weights
    from benchmarks.kernel.mext_paper_sweep import _make_runners
    from benchmarks.kernel.timing import fi_autotuned, timed_detail
    from quant.flashinfer_integration import mxfp4_linear_adapter, sm100_fold_candidates

    from .reference import metrics

    arms = [
        "fi_nvfp4_cutedsl",
        "mext_r1",
        "fi_mxfp4_mxfp4_stock",
        "resinfer_mxfp4_mext_r1_tuned",
    ]
    snapshot_saved = False
    for name, w in weights.items():
        t = time.time()
        whash = digest(w)
        packed = quantize_mext_weights(w, arms)
        for (index, step, label), x in activations.items():
            if label != name:
                continue
            reference = x.float() @ w.float().T
            # Validate FP32 reference on a fixed output slice using FP64.
            ref64 = x.double() @ w[:128].double().T
            nmse = metrics(reference[:, :128], ref64)["nmse"]
            assert nmse is not None and nmse < 1e-9
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
            runners = _make_runners(x, packed, arms)
            for arm in arms:
                fn, inputs = runners[arm]
                fi_autotuned(fn, inputs, m_bucket=1)
                output = fn(*inputs).clone()
                again = fn(*inputs).clone()
                assert torch.equal(output, again)
                tactic = None
                if arm == "mext_r1":
                    tactic = sm100_fold_candidates.last_tactic_tile()
                    assert tactic is not None
                if arm == "resinfer_mxfp4_mext_r1_tuned":
                    tactic = mxfp4_linear_adapter.last_tactic_tile()
                    assert tactic is not None
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
                    "input_sha256": digest(x),
                    "tactic": tactic,
                    "repeat_equal": True,
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
