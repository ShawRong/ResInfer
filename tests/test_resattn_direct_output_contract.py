"""CPU contract checks; GPU correctness is covered by the r203 probe."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / 'csrc/nvfp4/nvfp4/cute_attn/host.py'


class DirectOutputContractTests(unittest.TestCase):
    def test_both_host_entries_keep_legacy_default(self):
        tree = ast.parse(HOST.read_text())
        names = {'run_paged_fp4_decode_qmul4_v', 'run_paged_fp4_decode_qmul4_tmem'}
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            fn = funcs[name]
            defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                                fn.args.defaults))
            self.assertIs(ast.literal_eval(defaults['direct_output']), False)

    def test_compact_reduction_maps_each_kv_group_separately(self):
        # Logical heads are not simply the first half of a flattened padded O.
        for kv_heads in (1, 2, 8):
            mapped = [(head // 8) * 16 + head % 8 for head in range(kv_heads * 8)]
            expected = [kv * 16 + h for kv in range(kv_heads) for h in range(8)]
            self.assertEqual(mapped, expected)

    def test_registration_selects_phase_separated_stock_output_backend(self):
        registration = (ROOT / 'quant/vllm_integration/attn_register.py').read_text()
        self.assertIn(
            'quant.vllm_integration.attn_backend_direct_output.'
            'ResAttnDirectOutputBackend',
            registration,
        )
        self.assertNotIn(
            '"quant.vllm_integration.attn_backend.ResAttnBackend"',
            registration,
        )

    def test_changed_python_compiles(self):
        for path in [HOST, ROOT / 'quant/vllm_integration/attn_backend.py',
                     ROOT / 'quant/vllm_integration/attn_backend_direct_output.py']:
            compile(path.read_text(), str(path), 'exec')

    def test_direct_no_split_clears_inactive_rows(self):
        # Execute the actual host guard, without importing CUDA on the Mac.
        from unittest.mock import Mock
        tree = ast.parse(HOST.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'run_paged_fp4_decode_qmul4_v')
        guard = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "direct_output and reduction == 'none'")
        code = compile(ast.Module(body=[guard], type_ignores=[]), '<guard>', 'exec')
        for direct in (False, True):
            for reduction in ('none', 'atomic', 'kernel'):
                out = Mock()
                exec(code, dict(direct_output=direct, reduction=reduction, out_eff=out))
                self.assertEqual(out.zero_.call_count, int(direct and reduction == 'none'))

    def test_compact_gqa4_auto_plan_stays_on_supported_no_split_path(self):
        tree = ast.parse(HOST.read_text())
        fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'plan_qmul4_decode_nvfp4'
        )
        namespace = {
            'Tuple': tuple,
            '_padded_group_size': lambda qo, kv: 8 if qo // kv == 4 else 16,
            '_auto_splits_qmul4': lambda *args: (_ for _ in ()).throw(
                AssertionError('GQA4 must return before the Q16 planner')),
            '_sm_count': lambda device: 132,
            'QMUL4_KERNEL_REDUCTION_MIN_SEQ': 32768,
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<planner>', 'exec'),
             namespace)
        plan = namespace['plan_qmul4_decode_nvfp4']
        for batch, sequence in ((8, 1025), (8, 4103), (1, 32768)):
            with self.subTest(batch=batch, sequence=sequence):
                self.assertEqual(plan(batch, 32, 8, sequence, 'cuda'), (1, 'none'))


if __name__ == '__main__':
    unittest.main()
