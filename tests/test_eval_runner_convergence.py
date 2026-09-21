from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.gpqa_deterministic import gold_index, ordered_answers


REPO = Path(__file__).resolve().parent.parent


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "resinfer_lcb_driver", REPO / "benchmarks" / "lcb_driver.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_pythonpath_contains_both_package_roots(tmp_path):
    driver = _load_driver()
    args = SimpleNamespace(gpu=None, meff_flush_every=32, mext_max_m=64)
    arm = {"kind": "residue", "abs_path": "/checkpoint", "mext": "on"}
    with patch.dict(os.environ, {}, clear=True):
        env = driver.build_server_env(
            arm, args, tmp_path, tmp_path / "probe", tmp_path / "events",
            tmp_path / "enabled")
    paths = env["PYTHONPATH"].split(os.pathsep)
    assert paths[:3] == [
        str(tmp_path / "probe"),
        str(REPO),
        str(REPO / "csrc" / "nvfp4"),
    ]


def test_launch_server_reports_early_child_exit(tmp_path):
    driver = _load_driver()
    started = __import__("time").monotonic()
    with pytest.raises(RuntimeError, match="exited with code 7"):
        driver.launch_server(
            [sys.executable, "-c", "raise SystemExit(7)"],
            os.environ.copy(), tmp_path, 9, 30)
    assert __import__("time").monotonic() - started < 10
    driver.teardown()


def test_gpqa_order_is_stable_across_fresh_processes():
    question = "Which answer is correct?"
    code = (
        "from benchmarks.gpqa_deterministic import gold_index; "
        f"print(gold_index({question!r}))"
    )
    outputs = [
        subprocess.check_output([sys.executable, "-c", code], cwd=REPO, text=True)
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1] == f"{gold_index(question)}\n"
    answers, index = ordered_answers({
        "Question": question,
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong 1",
        "Incorrect Answer 2": "wrong 2",
        "Incorrect Answer 3": "wrong 3",
    })
    assert answers[index] == "correct"


def test_explicit_sitecustomize_patch_failure_stops_python(tmp_path):
    (tmp_path / "sitecustomize.py").write_text((REPO / "sitecustomize.py").read_text())
    package = tmp_path / "quant" / "vllm_integration"
    package.mkdir(parents=True)
    (tmp_path / "quant" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "patch.py").write_text(
        "def patch_vllm_modelopt(model_dir=None): return False\n"
        "def is_patched(): return False\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["RESIDUE_PATCH_VLLM"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "print('continued')"], env=env,
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "continued" not in result.stdout
    assert "patch was not installed" in result.stderr


def test_scorer_compat_encodes_text_for_xxhash(tmp_path):
    (tmp_path / "xxhash.py").write_text(
        "class Hash:\n"
        "  def hexdigest(self): return 'ok'\n"
        "def xxh64(data=b'', *args, **kwargs):\n"
        "  if isinstance(data, str): raise TypeError('bytes required')\n"
        "  return Hash()\n")
    compat = REPO / "envs" / "lighteval" / "compat"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(compat), str(tmp_path)])
    result = subprocess.run(
        [sys.executable, "-c",
         "import xxhash; print(xxhash.xxh64('text').hexdigest())"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_offline_bootstrap_rejects_archive_links(tmp_path):
    bootstrap_path = REPO / "envs" / "lighteval" / "bootstrap_offline.py"
    spec = importlib.util.spec_from_file_location("bootstrap_offline", bootstrap_path)
    assert spec and spec.loader
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        link = tarfile.TarInfo("nltk_data/tokenizers/punkt")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        bundle.addfile(link)
    with pytest.raises(RuntimeError, match="unsupported archive member type"):
        bootstrap._safe_extract(archive, tmp_path / "out")
