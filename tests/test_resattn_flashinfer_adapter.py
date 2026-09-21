"""Run in the vLLM 0.26 GPU Runtime; mocks test routing, not kernel accuracy."""
from dataclasses import fields
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_RUNTIME = all(importlib.util.find_spec(m) for m in ("torch", "vllm"))
if HAS_RUNTIME:
    import torch
    from quant.vllm_integration import attn_backend_flashinfer as adapter


@unittest.skipUnless(HAS_RUNTIME, "requires the pinned vLLM Runtime")
class AdapterTests(unittest.TestCase):
    def metadata(self, n=2, p=3):
        args = {f.name: None for f in fields(adapter.FlashInferMetadata)}
        args.update(num_actual_tokens=n+p, slot_mapping=torch.arange(n+p),
                    q_data_type_prefill=torch.float8_e4m3fn,
                    q_data_type_decode=torch.float8_e4m3fn,
                    num_decodes=n, num_decode_tokens=n,
                    num_prefills=int(p > 0), num_prefill_tokens=p,
                    causal=True, use_cascade=False)
        args['prefill'] = adapter.TRTLLMPrefill(
            torch.tensor([[6, 7]], dtype=torch.int32),
            torch.tensor([p], dtype=torch.int32),
            torch.tensor([0, p], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32), p, p,
        ) if p else None
        return adapter.FlashInferMetadata(**args)

    def test_raw_q_capability_overrides_parent_instance_value(self):
        def fake_init(obj):
            obj.supports_quant_query_input = True
            obj.is_kvcache_nvfp4 = True
            obj.head_size, obj.num_heads, obj.num_kv_heads = 128, 64, 8
            obj.alibi_slopes = obj.sinks = obj.logits_soft_cap = None
            obj.kv_sharing_target_layer_name = None
            obj.window_left = -1
        with patch.object(adapter.FlashInferImpl, '__init__', fake_init), \
             patch.object(adapter, '_register_decode_warmup'):
            impl = adapter.ResAttnFlashInferImpl()
        self.assertFalse(impl.supports_quant_query_input)
        self.assertFalse(impl.fused_output_quant_supported(None))

    def test_constructor_accepts_supported_gqa_ratios(self):
        def fake_init(obj):
            obj.supports_quant_query_input = True
            obj.is_kvcache_nvfp4 = True
            obj.head_size, obj.num_kv_heads = 128, 8
            obj.alibi_slopes = obj.sinks = obj.logits_soft_cap = None
            obj.kv_sharing_target_layer_name = None
            obj.window_left = -1

        for gqa in (4, 6, 8):
            with self.subTest(gqa=gqa):
                def init_with_heads(obj, gqa=gqa):
                    fake_init(obj)
                    obj.num_heads = gqa * obj.num_kv_heads

                with patch.object(
                    adapter.FlashInferImpl, '__init__', init_with_heads
                ), patch.object(adapter, '_register_decode_warmup'):
                    impl = adapter.ResAttnFlashInferImpl()
                self.assertFalse(impl.supports_quant_query_input)

    def test_constructor_rejects_unsupported_gqa_ratio(self):
        for num_heads, num_kv_heads in ((40, 8), (47, 8), (48, 0)):
            with self.subTest(num_heads=num_heads, num_kv_heads=num_kv_heads):
                def fake_init(obj):
                    obj.supports_quant_query_input = True
                    obj.is_kvcache_nvfp4 = True
                    obj.head_size = 128
                    obj.num_heads, obj.num_kv_heads = num_heads, num_kv_heads
                    obj.alibi_slopes = obj.sinks = obj.logits_soft_cap = None
                    obj.kv_sharing_target_layer_name = None
                    obj.window_left = -1

                with patch.object(adapter.FlashInferImpl, '__init__', fake_init), \
                     patch.object(adapter, '_register_decode_warmup'), \
                     self.assertRaisesRegex(NotImplementedError, 'GQA4/6/8'):
                    adapter.ResAttnFlashInferImpl()

    def test_stock_owns_cache_contract_and_writer(self):
        backend = adapter.ResAttnFlashInferBackend
        self.assertFalse(backend.forward_includes_kv_cache_update)
        self.assertIs(backend.get_kv_cache_shape,
                      adapter.FlashInferBackend.get_kv_cache_shape)
        self.assertIs(adapter.ResAttnFlashInferImpl.do_kv_cache_update,
                      adapter.FlashInferImpl.do_kv_cache_update)

    def make_builder(self):
        obj = adapter.ResAttnFlashInferMetadataBuilder.__new__(
            adapter.ResAttnFlashInferMetadataBuilder)
        obj._residue_offsets = {}
        obj._residue_max_reqs = 8
        obj._cg_capture_seq_len = 40960
        return obj

    def test_dense_page_table_zero_copy_and_persistent_offsets(self):
        builder = self.make_builder()
        cam = SimpleNamespace(
            block_table_tensor=torch.tensor([[7, 3, 0], [5, 2, 9], [6, 7, 0]],
                                            dtype=torch.int32),
            seq_lens=torch.tensor([129, 300, 3], dtype=torch.int32),
            max_seq_len=300,
        )
        # cam deliberately has no seq_lens_cpu: no D2H staging is needed.
        with patch.object(adapter.FlashInferMetadataBuilder, 'build',
                          return_value=self.metadata(p=0)):
            first = builder.build(0, cam)
            second = builder.build(0, cam)
            captured = builder.build_for_cudagraph_capture(cam)
        dec = first.residue_decode
        self.assertEqual(dec.kv_indices.data_ptr(), cam.block_table_tensor.data_ptr())
        self.assertEqual(dec.kv_indptr.tolist(), [0, 3, 6])
        self.assertEqual(dec.kv_indptr.data_ptr(), second.residue_decode.kv_indptr.data_ptr())
        self.assertEqual(captured.residue_decode.max_seq_len, 40960)
        self.assertIsNone(first.stock)

    def test_prefill_and_mixed_builds_reuse_stock_without_decode_descriptor(self):
        builder = self.make_builder()
        cam = object()  # Proves the fallback never reads decode page metadata.
        for n in (0, 2):
            original = self.metadata(n=n)
            with self.subTest(num_decodes=n), patch.object(
                    adapter.FlashInferMetadataBuilder, 'build',
                    return_value=original):
                md = builder.build(0, cam)
            self.assertIs(md.stock, original)
            self.assertIsNone(md.residue_decode)

    def test_pure_prefill_and_mixed_forward_delegate_whole_step_to_stock(self):
        impl = adapter.ResAttnFlashInferImpl.__new__(adapter.ResAttnFlashInferImpl)
        for n in (0, 2):
            with self.subTest(num_decodes=n):
                p = 3
                q = torch.randn(n+p, 64, 128, dtype=torch.bfloat16)
                k = torch.randn(n+p, 8, 128, dtype=torch.bfloat16)
                output = torch.full_like(q, -1)
                stock = self.metadata(n=n, p=p)
                md = SimpleNamespace(num_prefill_tokens=p, stock=stock,
                                     residue_decode=None)

                def stock_forward(obj, layer, query, key, value, cache, meta, out):
                    self.assertIs(query, q)
                    self.assertIs(key, k)
                    self.assertIs(meta, stock)
                    out.fill_(2)
                    return out

                with patch.object(adapter.FlashInferImpl, 'forward', stock_forward), \
                     patch.object(impl, '_decode_forward') as decode, \
                     patch.object(impl, 'do_kv_cache_update') as writer:
                    result = impl.forward(object(), q, k, k, object(), md, output)
                writer.assert_not_called()
                decode.assert_not_called()
                self.assertIs(result, output)
                self.assertTrue((output == 2).all())

    def test_pure_decode_uses_resattn_without_stock_dispatch(self):
        impl = adapter.ResAttnFlashInferImpl.__new__(adapter.ResAttnFlashInferImpl)
        q = torch.randn(2, 64, 128, dtype=torch.bfloat16)
        k = torch.randn(2, 8, 128, dtype=torch.bfloat16)
        output = torch.full_like(q, -1)
        residue = object()
        md = SimpleNamespace(num_prefill_tokens=0, stock=None,
                             residue_decode=residue)

        def decode(query, key, value, pool, meta, out):
            self.assertIs(query, q)
            self.assertIs(meta, residue)
            out.fill_(1)

        with patch.object(adapter.FlashInferImpl, 'forward') as stock_forward, \
             patch.object(impl, '_pool_for', return_value=object()), \
             patch.object(impl, '_decode_forward', side_effect=decode), \
             patch.object(impl, 'do_kv_cache_update') as writer:
            result = impl.forward(object(), q, k, k, object(), md, output)
        writer.assert_not_called()
        stock_forward.assert_not_called()
        self.assertIs(result, output)
        self.assertTrue((output == 1).all())

    def test_prefill_build_never_touches_decode_pages(self):
        with patch.object(adapter.FlashInferMetadataBuilder, 'build',
                          return_value=self.metadata(n=0)):
            md = self.make_builder().build(0, object())
        self.assertIsNone(md.residue_decode)

    def test_fp8_query_rejected_not_dequantized(self):
        impl = adapter.ResAttnFlashInferImpl.__new__(adapter.ResAttnFlashInferImpl)
        q = torch.zeros(1, 64, 128).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, 'original BF16 Q'):
            impl.forward(None, q, None, None, None, object(),
                         torch.empty(1, 64, 128, dtype=torch.bfloat16))


class SourceTests(unittest.TestCase):
    def test_module_compiles_without_runtime(self):
        path = Path(__file__).resolve().parents[1] / 'quant/vllm_integration/attn_backend_flashinfer.py'
        compile(path.read_text(), str(path), 'exec')


if __name__ == '__main__':
    unittest.main()
