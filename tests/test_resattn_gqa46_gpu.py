"""GPU canary for compact GQA4/GQA6 direct-NVFP4 split reduction.

Run on one explicitly selected Blackwell GPU with the matching nvfp4 extension:
    CUDA_VISIBLE_DEVICES=<idle GPU> python -m unittest discover -s tests \
        -p test_resattn_gqa46_gpu.py -v
"""
from __future__ import annotations

import math
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch


def _unpack_output(data: torch.Tensor, scales: torch.Tensor,
                   heads: int, dim: int) -> torch.Tensor:
    """Decode the vLLM 128x4 swizzled SF layout for test-only comparison."""
    batch = data.shape[0]
    bytes_ = data.reshape(batch, heads, dim // 2).to(torch.int32)
    nibbles = torch.stack((bytes_ & 15, bytes_ >> 4), dim=-1).reshape(
        batch, heads, dim)
    lut = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.),
                       device=data.device)
    values = lut[nibbles & 7] * torch.where(
        (nibbles & 8) == 0, 1., -1.)
    blocks_per_head = dim // 16
    k_idx = torch.arange(heads * blocks_per_head, device=data.device)
    num_k_tiles = math.ceil(heads * dim / 64)
    rows = torch.arange(batch, device=data.device)
    sf_offsets = (
        (rows[:, None] // 128 * num_k_tiles + k_idx[None, :] // 4) * 512
        + rows[:, None] % 32 * 16
        + (rows[:, None] // 32) % 4 * 4
        + k_idx[None, :] % 4
    )
    sf = scales.reshape(-1)[sf_offsets].float().reshape(
        batch, heads, blocks_per_head)
    return values.reshape(batch, heads, blocks_per_head, 16).mul_(
        sf.unsqueeze(-1)).reshape(batch, heads, dim)


@unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(),
                     "requires a Blackwell GPU")
class GQA46FusedReductionCanary(unittest.TestCase):
    def test_split_vs_no_split_and_graph_replay(self):
        from nvfp4.cute_attn import host
        from nvfp4.cute_attn.kv_layout import NvFp4PagedKVPool

        torch.manual_seed(17)
        device = torch.device("cuda:0")
        for gqa in (4, 6):
            with self.subTest(gqa=gqa):
                batch, kv_heads, dim, seq = 1, 8, 128, 4103
                heads = kv_heads * gqa
                width = math.ceil(seq / 128)
                pages = batch * width
                shape = (pages, kv_heads, 128, dim // 2)
                k_data = torch.randint(0, 256, shape, dtype=torch.uint8,
                                       device=device)
                v_data = torch.randint(0, 256, shape, dtype=torch.uint8,
                                       device=device)
                sf = torch.full((pages, kv_heads, 128, dim // 16), .125,
                                dtype=torch.float8_e4m3fn, device=device)
                pool = NvFp4PagedKVPool(
                    num_kv_heads=kv_heads, head_dim=dim,
                    k_data=k_data, v_data=v_data, k_sf=sf, v_sf=sf,
                    v_staging=torch.empty(0, dtype=torch.bfloat16,
                                          device=device),
                    page_size=128, layout_profile="trtllm",
                    k_global_scale=1., v_global_scale=1.,
                )
                indices = torch.randperm(pages, device=device).to(torch.int32)
                indptr = torch.arange(batch + 1, device=device,
                                      dtype=torch.int32) * width
                seq_lens = torch.tensor((seq,), device=device,
                                        dtype=torch.int32)
                q = torch.randn(batch, heads, dim, device=device,
                                dtype=torch.bfloat16)
                o_scale_inv = torch.ones(1, device=device, dtype=torch.float32)
                auto_plan = host.plan_qmul4_decode_nvfp4(
                    batch, heads, kv_heads, seq, device, residue_fold=True)
                self.assertEqual(auto_plan, (4, "gmem_fused"))

                def run(plan, workspace=None, out=None, block_scale=None):
                    return host.run_paged_fp4_decode_qmul4_tmem_nvfp4(
                        q, pool, v_data, sf, indices, indptr, seq_lens,
                        dim ** -.5, o_scale_inv, out=out,
                        output_block_scale=block_scale, max_seq_len=seq,
                        k_sf_mode="linear", value_scale=1.,
                        reduction_workspace=workspace, plan=plan,
                        residue_fold=True,
                    )

                base_data, base_sf = run((1, "none"))
                split_data, split_sf = run(auto_plan)
                torch.cuda.synchronize()
                base = _unpack_output(base_data, base_sf, heads, dim)
                split = _unpack_output(split_data, split_sf, heads, dim)
                self.assertTrue(torch.isfinite(split).all())
                cosine = torch.nn.functional.cosine_similarity(
                    base.reshape(1, -1), split.reshape(1, -1)).item()
                self.assertGreater(cosine, .97, f"GQA{gqa}: {cosine=}")

                workspace = host.allocate_qmul4_gmem_reduction_workspace(
                    auto_plan[0], batch, kv_heads * (8 if gqa == 4 else 16),
                    dim, device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(auto_plan, workspace, split_data, split_sf)
                for _ in range(2):
                    graph.replay()
                torch.cuda.synchronize()
                replay = _unpack_output(split_data, split_sf, heads, dim)
                replay_cosine = torch.nn.functional.cosine_similarity(
                    base.reshape(1, -1), replay.reshape(1, -1)).item()
                self.assertGreater(replay_cosine, .97)
                self.assertTrue(torch.count_nonzero(workspace[3]) == 0)


if __name__ == "__main__":
    unittest.main()
