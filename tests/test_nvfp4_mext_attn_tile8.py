from __future__ import annotations

import importlib.util
import unittest

try:
    import torch
except ImportError:  # Keep CPU-only source checks collectable.
    torch = None

HAS_NVFP4 = importlib.util.find_spec("nvfp4") is not None

@unittest.skipUnless(torch is not None and HAS_NVFP4, "requires nvfp4 Runtime")
class MExtAttnTile8Test(unittest.TestCase):
    def test_attn_tile8_layout_aliases(self) -> None:
        from nvfp4.mext import _resolve_mext_r1_layout_mode

        for name in (
            "attn_tile8",
            "attention_tile8",
            "qk_tile8",
            "gqa_tile8",
        ):
            self.assertEqual(_resolve_mext_r1_layout_mode(name), 4)
    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_attn_tile8_reuses_tile8_data_and_expands_sf_per_group(self) -> None:
        import nvfp4
        from nvfp4.cute_attn.kv_layout import sf_atom_gather

        if not hasattr(torch.ops.nvfp4, "scaled_fp4_quant_mext_r1"):
            self.skipTest("nvfp4 extension lacks scaled_fp4_quant_mext_r1")

        torch.manual_seed(20260904)
        m, k = 16, 128  # two logical GQA=8 groups
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        input_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        tile8_data, tile8_sf = nvfp4.scaled_fp4_quant_mext_r1(
            x, input_scale, elts_mode=16, layout_mode="tile8"
        )
        attn_data, attn_sf = nvfp4.scaled_fp4_quant_mext_r1(
            x, input_scale, elts_mode=16, layout_mode="attn_tile8"
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(attn_data, tile8_data))
        self.assertEqual(tile8_sf.numel(), 128 * (k // 16))
        self.assertEqual(attn_sf.numel(), (m // 8) * 128 * (k // 16))

        tile8_linear = sf_atom_gather(
            tile8_sf.reshape(1, -1), 128, k // 16
        )[0]
        attn_linear = sf_atom_gather(
            attn_sf.reshape(m // 8, -1), 128, k // 16
        )
        for group in range(m // 8):
            self.assertTrue(
                torch.equal(
                    attn_linear[group, :16],
                    tile8_linear[group * 16 : (group + 1) * 16],
                )
            )
            self.assertEqual(
                int(torch.count_nonzero(attn_linear[group, 16:])), 0
            )
    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_resattn_host_reuses_mext_contract_for_multiple_groups(
        self,
    ) -> None:
        import nvfp4
        from nvfp4.cute_attn import host

        if not hasattr(torch.ops.nvfp4, "scaled_fp4_quant_mext_r1"):
            self.skipTest("nvfp4 extension lacks scaled_fp4_quant_mext_r1")

        torch.manual_seed(20260905)
        batch, h_kv, g, d = 2, 2, 8, 128
        q = torch.randn(
            batch, h_kv * g, d, dtype=torch.bfloat16, device="cuda"
        )
        ref_data, ref_sf = nvfp4.scaled_fp4_quant_mext_r1(
            q.reshape(batch * h_kv * g, d),
            torch.ones(1, dtype=torch.float32, device="cuda"),
            elts_mode="auto",
            layout_mode="attn_tile8",
        )
        got = host._quantize_q_residue_cuda(q, h_kv)
        self.assertIsNotNone(got)
        assert got is not None
        got_data, got_sf = got
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(got_data.reshape(batch * h_kv * 16, d // 2), ref_data)
        )
        self.assertTrue(
            torch.equal(got_sf.view(torch.uint8), ref_sf.view(torch.uint8))
        )


if __name__ == "__main__":
    unittest.main()
