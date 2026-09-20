"""CPU checks for attention A/B config, not GPU/backend acceptance."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "resattn_e2e_plan",
    Path(__file__).resolve().parents[1] / "benchmarks/resattn_e2e_plan.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class ResAttnE2EPlanTests(unittest.TestCase):
    def plan(self, config=None, **kw):
        cfg = dict(hidden_size=5120, head_dim=128, num_attention_heads=64,
                   num_key_value_heads=8, max_position_embeddings=40960)
        cfg.update(config or {})
        args = dict(python="python", gpu="3", input_len=32768, output_len=512,
                    concurrency=64, max_model_len=40960)
        args.update(kw)
        return _module.build_plan("/checkpoint", cfg, **args)

    def test_identical_model_and_common_cli(self):
        p = self.plan()
        a, b = [p["arms"][k]["server_argv"] for k in ("trt", "resattn")]
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual((a[-1], b[-1]), ("FLASHINFER", "CUSTOM"))
        self.assertEqual(p["status"], "PLAN_ONLY_NOT_RUN")

    def test_clean_attention_only_environment(self):
        with patch.dict("os.environ", {"RESIDUE_PATCH_VLLM": "0",
                                      "RESIDUE_ATTN_DISABLE_QK_RESIDUE": "1",
                                      "RESIDUE_ATTN_PREFILL_BF16": "1",
                                      "NVFP4_TEST_OVERRIDE": "1"}):
            p = self.plan()
        for arm in p["arms"].values():
            for name in ("RESIDUE_PATCH_VLLM", "RESIDUE_ATTN_DISABLE_QK_RESIDUE",
                         "RESIDUE_ATTN_PREFILL_BF16", "NVFP4_TEST_OVERRIDE"):
                self.assertIn(name, arm["env_unset"])
                self.assertNotIn(name, arm["env_set"])
        self.assertNotIn("RESIDUE_ATTN", p["arms"]["trt"]["env_set"])

    def test_reject_nonresidue_geometry(self):
        with self.assertRaisesRegex(ValueError, "GQA8"):
            self.plan(config={"num_attention_heads": 32})

    def test_no_implicit_context_extension(self):
        with self.assertRaisesRegex(ValueError, "context limit"):
            self.plan(input_len=65536, max_model_len=66048)
        with self.assertRaisesRegex(ValueError, "input"):
            self.plan(max_model_len=32768)

    def test_eager_smoke_does_not_enable_graph(self):
        p = self.plan(eager=True)
        for arm in p["arms"].values():
            self.assertIn("--enforce-eager", arm["server_argv"])
            self.assertNotIn("RESIDUE_ATTN_CUDAGRAPH", arm["env_set"])

    def test_tp_keeps_per_rank_residue_geometry(self):
        self.plan(tp=2, gpu="2,3")
        for kw in ({"tp": 2}, {"tp": 2, "gpu": "3,3"},
                   {"tp": 3, "gpu": "2,3,4"}):
            with self.assertRaises(ValueError):
                self.plan(**kw)

    def test_large_batch_request_count_not_capped_at_256(self):
        argv = self.plan(concurrency=128)["benchmark_argv"]
        self.assertEqual(argv[argv.index("--num-prompts") + 1], "1024")
        # vLLM0.26 uses [len*(1-ratio), len*(1+ratio)]; zero is fixed length.
        self.assertEqual(argv[argv.index("--random-range-ratio") + 1], "0.0")

    def test_offline_cubins_same_for_both_arms(self):
        p = self.plan(offline_cubin_dir="/verified/cubins")
        for arm in p["arms"].values():
            self.assertEqual(arm["env_set"]["VLLM_HAS_FLASHINFER_CUBIN"], "1")
            self.assertEqual(arm["env_set"]["FLASHINFER_NO_DOWNLOAD"], "1")
            self.assertEqual(arm["env_set"]["FLASHINFER_CUBIN_DIR"], "/verified/cubins")

    def test_fixed_cache_and_fusion_match_between_arms(self):
        import json
        p = self.plan(kv_cache_bytes=85899345920, attn_quant_fusion=True)
        for arm in p["arms"].values():
            argv = arm["server_argv"]
            self.assertEqual(argv[argv.index("--kv-cache-memory") + 1], "85899345920")
            config = json.loads(argv[argv.index("--compilation-config") + 1])
            self.assertTrue(config["pass_config"]["fuse_attn_quant"])
        with self.assertRaises(ValueError):
            self.plan(eager=True, attn_quant_fusion=True)


if __name__ == "__main__":
    unittest.main()
