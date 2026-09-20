"""Execute the actual host selection block without importing CUDA packages."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'csrc/nvfp4/nvfp4/cute_attn/host.py').read_text())
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
BODY = FUNCS['run_paged_fp4_decode_qmul4_v'].body
START = next(i for i, n in enumerate(BODY)
             if isinstance(n, ast.If) and ast.unparse(n.test) == 'plan is None')
BLOCK = compile(ast.Module(body=BODY[START:START + 2], type_ignores=[]),
                '<actual host selection block>', 'exec')


class PlanPassthroughTests(unittest.TestCase):
    def run_block(self, plan=None, max_len=8192, force=False, selector=None,
                  batch=3, heads=8):
        selector = selector or Mock(return_value=(4, 'atomic'))
        seq = Mock()
        seq.max.return_value.item.return_value = 8192
        env = dict(plan=plan, max_seq_len=max_len, force_single_split=force,
                   seq_lens=seq, B=batch, H=heads, q=SimpleNamespace(device='cuda:0'),
                   _auto_splits_qmul4=selector)
        exec(BLOCK, env)
        return (env['splits'], env['reduction']), selector, seq

    def test_supplied_plan_neither_replans_nor_reads_device_length(self):
        selected = (8, 'kernel')
        result, selector, seq = self.run_block(plan=selected, max_len=0)
        self.assertEqual(result, selected)
        selector.assert_not_called()
        seq.max.assert_not_called()

    def test_standalone_default_retains_original_selector(self):
        result, selector, seq = self.run_block()
        self.assertEqual(result, (4, 'atomic'))
        selector.assert_called_once_with(3, 8, 8192, 'cuda:0')
        seq.max.assert_not_called()

    def test_unknown_length_retains_original_fallback(self):
        result, selector, seq = self.run_block(max_len=0)
        self.assertEqual(result, (4, 'atomic'))
        selector.assert_called_once_with(3, 8, 8192, 'cuda:0')
        seq.max.return_value.item.assert_called_once()

    def test_force_single_split_keeps_precedence(self):
        for plan in (None, (8, 'kernel')):
            result, _, _ = self.run_block(plan=plan, force=True)
            self.assertEqual(result, (1, 'none'))

    def test_actual_policy_matrix_unchanged_when_forwarded(self):
        # Use the existing policy's source, not a reimplementation of its
        # thresholds. Only hardware SM count is supplied as a CPU fixture.
        names = ('_split_reduction', '_auto_splits_qmul4', 'plan_qmul4_decode')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                            names=[ast.alias(name='annotations')], level=0)] +
                            [FUNCS[n] for n in names], type_ignores=[])
        env = dict(os=os, _sm_count=lambda device: 148,
                   QMUL4_KERNEL_REDUCTION_MIN_SEQ=49152,
                   ATOMIC_REPRODUCIBLE_SPLITS=(1, 2))
        exec(compile(ast.fix_missing_locations(module), '<actual policy>', 'exec'), env)
        for mode in ('legacy', 'auto', 'deterministic'):
            with patch.dict(os.environ, {'RESIDUE_ATTN_REDUCTION': mode}):
                for batch in (1, 2, 3, 8, 32, 128):
                    for heads in (1, 8):
                        for length in (1, 1024, 1025, 4096, 4097, 8192,
                                       32768, 49151, 49152, 65536):
                            with self.subTest(mode=mode, batch=batch, heads=heads,
                                              length=length):
                                plan = env['plan_qmul4_decode'](batch, heads, length, 'cuda:0')
                                old, _, _ = self.run_block(max_len=length, batch=batch,
                                    heads=heads, selector=env['_auto_splits_qmul4'])
                                new, selector, _ = self.run_block(plan=plan, max_len=length,
                                                                 batch=batch, heads=heads)
                                self.assertEqual(old, new)
                                selector.assert_not_called()

    def test_tmem_entry_forwards_plan(self):
        fn = FUNCS['run_paged_fp4_decode_qmul4_tmem']
        call = fn.body[-1].value
        self.assertTrue(any(k.arg == 'plan' and isinstance(k.value, ast.Name)
                            and k.value.id == 'plan' for k in call.keywords))

    def test_backend_passes_the_same_plan_used_for_workspace(self):
        tree = ast.parse((ROOT / 'quant/vllm_integration/attn_backend.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'ResAttnImpl')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_decode_forward')
        start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == 'reduction_workspace')
        # Include planning and the host call, excluding the final output copy.
        block = compile(ast.Module(body=fn.body[start:-1], type_ignores=[]),
                        '<actual backend planning and call>', 'exec')
        for length in (0, 8192, 40960):
            with self.subTest(max_seq_len=length):
                plan = (8, 'kernel')
                host = SimpleNamespace(plan_qmul4_decode=Mock(return_value=plan),
                                       run_paged_fp4_decode_qmul4_tmem=Mock())
                owner = SimpleNamespace(num_heads=64, num_kv_heads=8, scale=0.1,
                                        _get_decode_reduction_workspace=Mock())
                env = dict(host=host, self=owner, n=3,
                    md=SimpleNamespace(max_seq_len=length, num_decodes=3,
                                       kv_indices=[], kv_indptr=[], seq_lens=[1, 2, 3]),
                    q_dec=SimpleNamespace(device='cuda:0'), kern_out=object(),
                    pool=SimpleNamespace(v_data=object(), v_sf=object(),
                                         k_global_scale=1., v_global_scale=1.),
                    direct_output=True, _qk_residue_for_geometry=lambda *args: True)
                exec(block, env)
                kwargs = host.run_paged_fp4_decode_qmul4_tmem.call_args.kwargs
                self.assertEqual(kwargs['max_seq_len'], length)
                if length:
                    host.plan_qmul4_decode.assert_called_once_with(3, 8, length, 'cuda:0')
                    self.assertIs(kwargs['plan'], plan)
                    self.assertIs(kwargs['reduction_workspace'],
                                  owner._get_decode_reduction_workspace.return_value)
                else:
                    host.plan_qmul4_decode.assert_not_called()
                    self.assertIsNone(kwargs['plan'])
                    self.assertIsNone(kwargs['reduction_workspace'])


if __name__ == '__main__':
    unittest.main()
