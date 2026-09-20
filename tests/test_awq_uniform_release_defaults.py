import ast
import sys
from pathlib import Path


def test_awq_export_defaults_to_uniform_r0125_at_4p5_bits(monkeypatch):
    from export import residue_export_flow

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_residue_model.py", "--model_path", "/tmp/local-model"],
    )
    args = residue_export_flow._parse_args()

    assert args.ratios == "0.125"
    assert args.budget == "4.5"


def test_gptq_fallback_uses_the_same_uniform_ratio():
    path = Path(__file__).parents[1] / "export/run_gptq_uniform_residue.py"
    tree = ast.parse(path.read_text())
    defaults = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--ratio"
        for keyword in node.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    ]

    assert defaults == [0.125]
