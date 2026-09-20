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

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_grouped_residue_q_selects_q8_or_q16(self) -> None:
        import nvfp4
        from nvfp4.cute_attn import host
        from nvfp4.cute_attn.kv_layout import sf_atom_gather, sf_atom_scatter

        if not hasattr(torch.ops.nvfp4, "nvfp4_quantize_q_residue"):
            self.skipTest("nvfp4 extension lacks grouped residue-Q producer")

        torch.manual_seed(20260920)
        batch, h_kv, d = 2, 3, 128
        for g in (4, 6, 8):
            with self.subTest(gqa=g):
                q = torch.randn(
                    batch, h_kv * g, d,
                    dtype=torch.bfloat16, device="cuda",
                )
                got = host._quantize_q_residue_cuda(q, h_kv)
                self.assertIsNotNone(got)
                assert got is not None
                got_data, got_sf = got

                # The established MExt GQA8 layout is the byte reference:
                # pad each logical group independently, then quantize it.
                padded = torch.zeros(
                    batch, h_kv, 8, d,
                    dtype=torch.bfloat16, device="cuda",
                )
                padded[:, :, :g] = q.view(batch, h_kv, g, d)
                ref_data, ref_sf = nvfp4.scaled_fp4_quant_mext_r1(
                    padded.reshape(batch * h_kv * 8, d),
                    torch.ones(1, dtype=torch.float32, device="cuda"),
                    elts_mode=16,
                    layout_mode="attn_tile8",
                )
                physical_g = 8 if g == 4 else 16
                if g == 4:
                    groups = batch * h_kv
                    ref_tiles = ref_data.reshape(groups, 16, d // 2)
                    ref_data = torch.cat(
                        (ref_tiles[:, :4], ref_tiles[:, 8:12]), dim=1
                    ).reshape(groups * physical_g, d // 2)
                    kg = d // 16
                    ref_linear = sf_atom_gather(
                        ref_sf.reshape(groups, -1), 128, kg
                    )
                    q8_linear = torch.zeros_like(ref_linear)
                    q8_linear[:, :4] = ref_linear[:, :4]
                    q8_linear[:, 4:8] = ref_linear[:, 8:12]
                    ref_sf = sf_atom_scatter(q8_linear).reshape(-1).view(
                        torch.float8_e4m3fn
                    )
                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(
                        got_data.reshape(
                            batch * h_kv * physical_g, d // 2
                        ),
                        ref_data,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        got_sf.view(torch.uint8), ref_sf.view(torch.uint8)
                    )
                )


if __name__ == "__main__":
    unittest.main()
