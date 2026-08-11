#!/usr/bin/env python3
'One-shot campaign exporter: produce every arm of an eval phase, gated.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))




























RECIPE = {
    # IDENTITY, not location. The location moves between machines; the
    # recipe does not. `dataset_sha` below is what the fingerprint actually
    # trusts -- this string is for humans and for error messages.
    "dataset_id": "llama-nemotron-post-training/calibration_sft_n1024",
    "calib_samples": 1024,
    "calib_seq_len": 2048,
    "seed": 42,
    "quant_fmt": "nvfp4",
    "quant_scope": "all",
    "calib_block_size": 8,
    "salience_metric": "activation_max",
}

# ── where things live: per-machine, and therefore NOT in this file ────────
# The campaign runs on at least two machines (locally to prove the chain,
# on the pod to produce the arms), so a path baked in here is wrong
# everywhere except where it was written. Worse, it used to be hashed into
# the recipe fingerprint, which made the fingerprint a machine id that
# always matched locally and could never match anywhere else -- invisible
# to any amount of local testing, by construction.
#
# Set these. `preflight` names the variable, not the missing file: on a pod
# "no such path" reads as a broken sync, and "RESIDUE_CAMPAIGN_BASE unset"
# does not.
DATASET_PATH = os.environ.get("RESIDUE_CAMPAIGN_DATASET", "")
BASE_MODEL = os.environ.get("RESIDUE_CAMPAIGN_BASE", "")
# The untouched model, before GPTQ. Kept separate from BASE_MODEL on
# purpose: conflating the two is what made every damage number in this
# project price the whole pipeline while reading as NVFP4's cost.
ORIG_BF16 = os.environ.get("RESIDUE_CAMPAIGN_ORIG_BF16", "")

_ENV_HELP = {
    "RESIDUE_CAMPAIGN_DATASET": "calibration .jsonl for " + RECIPE["dataset_id"],
    "RESIDUE_CAMPAIGN_BASE": "the GPTQ base every quantised arm is built from",
    "RESIDUE_CAMPAIGN_ORIG_BF16": "the model before GPTQ (reference arm)",
}


def _sha_file(path: str) -> str:
    'Implementation details.'
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha_model(path: str) -> str:
    "Head 8 MB + tail 8 MB + byte count over a model directory's tensors."
    h = hashlib.sha256()
    files = sorted(Path(path).glob("*.safetensors")) or sorted(Path(path).glob("*.bin"))
    for f in files:
        size = f.stat().st_size
        h.update(f"{f.name}:{size}".encode())
        with open(f, "rb") as fh:
            h.update(fh.read(8 << 20))
            if size > (16 << 20):
                fh.seek(-(8 << 20), 2)
                h.update(fh.read(8 << 20))
    return h.hexdigest()[:16]

# ── arm specs ─────────────────────────────────────────────────────────────
# ratio None  -> plain NVFP4, no residue, no extension (the matched floor)
# policy None -> uniform: indices derived from THIS calibration's activation_max
# policy path -> an external allocation (the solver's), injected
SPECS: dict[str, list[dict[str, Any]]] = {
    # P1 baseline + P3 ladder + P2 allocation, one recipe, one directory
    "p1_p4_4b": [
        {"arm": "plain",          "ratio": None},
        {"arm": "uniform_r0125",  "ratio": 0.125},
        {"arm": "uniform_r025",   "ratio": 0.25},
        {"arm": "uniform_r05",    "ratio": 0.5},
        {"arm": "uniform_r10",    "ratio": 1.0},
        # equal-budget solver arm; policy must be produced by auto_search and
        # its own provenance must name the calibration it was searched under.
        {"arm": "anchor", "ratio": 0.125, "policy": None,
         "note": "set policy= to the solver metadata; omitted arms are skipped"},
    ],
    # smallest thing that answers "is there anything to recover at all"
    "p0_floor_only": [
        {"arm": "plain", "ratio": None},
    ],
    # The solver's allocations at the SAME budgets as the uniform ladder, so
    # "uniform or solver" is finally an equal-budget question with one
    # calibration behind both sides. `ratio` here is only the budget LABEL --
    # it is not passed to calibration on this path (see build_arm); the whole
    # allocation, per-layer ratios included, comes from the policy file, and the
    # solver that produced it was run on the pinned recipe -- its driver's
    # defaults are c4 / num_samples=2 / max_sample_length=64, which is exactly
    # where the bad checkpoints came from.
    "solver_4b": [
        {"arm": "solver_b4p5", "ratio": 0.125,
         "policy": f"{REPO}/runs/solver_frontier_4b_seq2048/policy_metadata/"
                   "gradient_effective_bits_budget4p5_residue_metadata.json"},
        {"arm": "solver_b5", "ratio": 0.25,
         "policy": f"{REPO}/runs/solver_frontier_4b_seq2048/policy_metadata/"
                   "gradient_effective_bits_budget5_residue_metadata.json"},
        {"arm": "solver_b6", "ratio": 0.5,
         "policy": f"{REPO}/runs/solver_frontier_4b_seq2048/policy_metadata/"
                   "gradient_effective_bits_budget6_residue_metadata.json"},
        {"arm": "solver_b8", "ratio": 1.0,
         "policy": f"{REPO}/runs/solver_frontier_4b_seq2048/policy_metadata/"
                   "gradient_effective_bits_budget8_residue_metadata.json"},
    ],
}


def log(msg: str) -> None:
    print(f"[campaign {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def recipe_fingerprint(extra: dict | None = None) -> str:
    """Stable id for 'which calibration produced this'.

    The eval side compares fingerprints across arms; two arms that differ here
    are not comparable and `analyze` should say so rather than average them.
    """
    c = recipe_components(extra)
    # A fingerprint built from UNRESOLVED components is the worst possible
    # artifact: it is a valid-looking hash, and two arms exported with the
    # env unset would carry the SAME one, so `verify` would report "one
    # recipe across all arms" over two arms whose recipe is unknown. Refuse
    # to produce a value instead -- an obviously-broken output beats a
    # plausible one, every time.
    unresolved = [k for k, v in c.items() if v in ("UNRESOLVED", "UNSET")]
    if unresolved:
        return "UNRESOLVED(" + ",".join(sorted(unresolved)) + ")"
    return _fp_from(c)


_FP_CACHE: dict = {}


def recipe_components(extra: dict | None = None) -> dict:
    """Every field the fingerprint is made of, so a mismatch can be DIFFED.

    A gate that says "the fingerprint is wrong" sends the reader looking at
    everything; one that says "dataset_sha differs, dataset_id is the same"
    tells them in one line whether they synced wrong or are holding a
    different calibration file -- two situations with nothing in common
    except the symptom. Every expensive debugging session in this project
    began with a gate that knew more than it said.

    Digests, not paths: the same file on two machines must fingerprint the
    same, and two different files under one name must not.
    """
    c = dict(RECIPE)
    if DATASET_PATH and Path(DATASET_PATH).exists():
        c["dataset_sha"] = _FP_CACHE.setdefault(
            ("d", DATASET_PATH), _sha_file(DATASET_PATH))
    else:
        c["dataset_sha"] = "UNRESOLVED"
    if BASE_MODEL and Path(BASE_MODEL).is_dir():
        c["base_model_sha"] = _FP_CACHE.setdefault(
            ("m", BASE_MODEL), _sha_model(BASE_MODEL))
    else:
        c["base_model_sha"] = "UNRESOLVED"
    c["base_model_id"] = Path(BASE_MODEL).name if BASE_MODEL else "UNSET"
    if extra:
        c.update(extra)
    return c


def _fp_from(components: dict) -> str:
    blob = json.dumps(components, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def explain_fingerprint_mismatch(expected: dict, got: dict) -> list[str]:
    """Which COMPONENT differs -- the whole point of keeping them."""
    out = []
    for k in sorted(set(expected) | set(got)):
        e, g = expected.get(k, "<absent>"), got.get(k, "<absent>")
        if e != g:
            out.append(f"    {k}: expected {e!r}, got {g!r}")
    return out or ["    (no component differs -- compare the hashing itself)"]


# ── preflight: everything that can be wrong before a GPU is touched ───────
def preflight(spec_name: str) -> int:
    """Name every missing input at once. A miss prints MISSING, never nothing.

    Skipping a line makes 'absent' and 'fine' look identical, and absent is
    what this is hunting.
    """
    bad = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        print(f"  {'OK     ' if ok else 'MISSING'} {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            bad += 1

    print("=== inputs ===")
    for var, what in _ENV_HELP.items():
        check(f"env {var}", bool(os.environ.get(var)), what)
    check("base model", bool(BASE_MODEL) and Path(BASE_MODEL).is_dir(), BASE_MODEL
          or "set RESIDUE_CAMPAIGN_BASE")
    check("calibration dataset", bool(DATASET_PATH) and Path(DATASET_PATH).exists(),
          DATASET_PATH or "set RESIDUE_CAMPAIGN_DATASET")
    check("reference model (orig bf16)", bool(ORIG_BF16) and Path(ORIG_BF16).is_dir(),
          ORIG_BF16 or "set RESIDUE_CAMPAIGN_ORIG_BF16")

    print("=== packages (export venv) ===")
    for mod in ("torch", "modelopt.torch.quantization", "safetensors", "transformers"):
        try:
            __import__(mod)
            check(mod, True)
        except Exception as exc:  # noqa: BLE001
            check(mod, False, f"{type(exc).__name__}: {exc}")

    print("=== stages ===")
    try:
        from export.stage_adapters import (  # noqa: F401
            CheckQuantizerStateStage, ExportHFStage, InjectResidueMetadataStage,
            InspectQuantExportStage, LoadHFStage, ModelOptCalibrateStage,
            SnapshotQuantizerStateStage, StageContext,
        )
        check("export.stage_adapters (8 stages)", True)
        check("InspectQuantExportStage._assert_extended_widths",
              hasattr(InspectQuantExportStage, "_assert_extended_widths"),
              "the widening gate -- without it a bad export dies at serving time")
    except Exception as exc:  # noqa: BLE001
        check("export.stage_adapters", False, f"{type(exc).__name__}: {exc}")

    print("=== spec ===")
    arms = SPECS.get(spec_name)
    check(f"spec {spec_name!r}", arms is not None,
          f"known: {', '.join(sorted(SPECS))}")
    if arms:
        for a in arms:
            if a.get("policy") is not None and not Path(a["policy"]).exists():
                check(f"policy for arm {a['arm']}", False, a["policy"])
        print(f"  {len(arms)} arm(s): {', '.join(a['arm'] for a in arms)}")

    print("=== gpu ===")
    try:
        import torch
        check("cuda", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as exc:  # noqa: BLE001
        check("cuda", False, str(exc))

    print(f"\nrecipe fingerprint: {recipe_fingerprint()}")
    for k, v in sorted(recipe_components().items()):
        print(f"    {k:20s} {v}")
    print("PREFLIGHT " + ("PASS" if not bad else f"FAIL -- {bad} missing"))
    return 0 if not bad else 3


# ── indices from the calibration itself ───────────────────────────────────
def metadata_from_quantizer_state(state_path: Path, ratio: float,
                                  block_size: int) -> dict:
    'Build injectable residue metadata out of a residue-mode calibration.'
    import torch
    st = torch.load(state_path, map_location="cpu", weights_only=False)["modules"]
    layers: dict[str, Any] = {}
    for name, mod in st.items():
        if not isinstance(mod, dict):
            continue
        iq = mod.get("input_quantizer")
        if not isinstance(iq, dict) or "_salient_indices" not in iq:
            continue
        idx = iq["_salient_indices"]
        idx = idx.tolist() if hasattr(idx, "tolist") else list(idx)
        if not idx:
            continue
        layers[name] = {
            "salient_indices": idx,
            "num_salient": len(idx),
            "runtime_mode": "extended_k",
        }
    return {"global": {"residue_ratio": ratio, "block_size": block_size},
            "layers": layers}


def find_quantizer_state(artifact_root: Path) -> Path:
    hits = sorted(artifact_root.rglob("quantizer_state.pt"))
    if not hits:
        raise FileNotFoundError(
            f"no quantizer_state.pt under {artifact_root} -- calibration did not "
            f"save its state, so indices cannot be derived from it")
    return hits[-1]


# ── one arm ───────────────────────────────────────────────────────────────
def build_arm(spec: dict, out_root: Path, gpu: str | None) -> dict:
    from export.stage_adapters import (
        CheckQuantizerStateStage, ExportHFStage, InjectResidueMetadataStage,
        InspectQuantExportStage, LoadHFStage, ModelOptCalibrateStage,
        SnapshotQuantizerStateStage, StageContext,
    )

    arm = spec["arm"]
    ratio = spec.get("ratio")
    policy = spec.get("policy")
    d = out_root / arm
    art = out_root / "_artifacts" / arm

    residue = ratio is not None
    
    
    
    #     every layer's salient set is derived from THIS calibration's
    
    
    
    
    
    
    
    
    
    
    quant_mode = "w4a4" if (policy or not residue) else "residue"

    log(f">>> {arm}  ratio={ratio}  mode={quant_mode}"
        + (f"  policy={policy}" if policy else ""))

    ctx = StageContext(case_dir=d)
    ctx = LoadHFStage(f"load_{arm}")(ctx, {
        "model_path": BASE_MODEL, "dtype": "bfloat16", "trust_remote_code": True})

    calib_params = {
        "model_path": BASE_MODEL,
        "dataset_id": RECIPE["dataset_id"],
        "dataset_path": DATASET_PATH,
        "calib_samples": RECIPE["calib_samples"],
        "calib_seq_len": RECIPE["calib_seq_len"],
        "seed": RECIPE["seed"],
        "quant_fmt": RECIPE["quant_fmt"],
        "quant_mode": quant_mode,
        "quant_scope": RECIPE["quant_scope"],
        "calib_block_size": RECIPE["calib_block_size"],
        "salience_metric": RECIPE["salience_metric"],
        "artifact_root": str(art),
        "verbose": False,
    }
    if residue and not policy:
        # Only the uniform path uses it; under w4a4 it is inert, and passing it
        # there would read as if it did something.
        calib_params["ratio"] = ratio
    ctx = ModelOptCalibrateStage(f"calib_{arm}")(ctx, calib_params)

    meta_path = None
    if residue:
        ctx = SnapshotQuantizerStateStage(f"snap_{arm}")(ctx, {})

        if policy:
            meta_path = Path(policy)
        else:
            # uniform: derive from THIS calibration, do not import an older
            # export's indices
            state = find_quantizer_state(art)
            meta = metadata_from_quantizer_state(
                state, ratio, RECIPE["calib_block_size"])
            if not meta["layers"]:
                raise AssertionError(
                    f"{arm}: residue-mode calibration produced no _salient_indices. "
                    f"Check quant_mode -- 'w4a4' records none and ignores ratio.")
            meta_path = out_root / "_indices" / f"{arm}.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta))
            log(f"    indices derived from this calibration: {len(meta['layers'])} layers, "
                f"num_salient[0]={next(iter(meta['layers'].values()))['num_salient']}")

        # Widening happens HERE, not in calibration (fail #2).
        ctx = InjectResidueMetadataStage(f"inject_{arm}")(ctx, {
            "residue_metadata_path": str(meta_path), "verbose": False})
        # Injection must not touch any amax; the assertion lives in the flow.
        ctx = CheckQuantizerStateStage(f"check_{arm}")(ctx, {})
        mism = ctx.metrics.get("quantizer_state_mismatches")
        if mism:
            raise AssertionError(f"{arm}: injection moved {mism} quantizer states")

    export_params = {
        "export_dir": str(d), "export_target": "vllm",
        "safe_serialization": True,
        "calib_block_size": RECIPE["calib_block_size"], "verbose": False,
    }
    if residue:
        # Fail #1: without this, layers/global are skipped and the export is
        # unservable while reporting success.
        export_params["save_kernel_metadata"] = True
        export_params["residue_ratio"] = ratio
    else:
        export_params["save_kernel_metadata"] = False
    ctx = ExportHFStage(f"export_{arm}")(ctx, export_params)

    # Fail #2's gate. Runs on every arm, including plain (where it asserts the
    # absence of a residue claim rather than its width).
    inspect_params: dict[str, Any] = {
        "export_dir": str(d),
        "require_finite_export_tensors": True,
        "forbid_salient_indices_in_safetensors": True,
    }
    if residue:
        inspect_params["expected_residue_metadata_min_layers"] = 1
        inspect_params["expected_extended_linear_dims_min"] = 1
    ctx = InspectQuantExportStage(f"inspect_{arm}")(ctx, inspect_params)

    prov = {
        "arm": arm,
        "base_model": BASE_MODEL,
        "residue_ratio": ratio,
        "policy": str(meta_path) if meta_path else None,
        "indices_source": (
            "n/a (plain)" if not residue
            else "external policy" if policy
            else "derived from this arm's own calibration"),
        "recipe": RECIPE,
        "recipe_fingerprint": recipe_fingerprint(),
        "recipe_components": recipe_components(),
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
    }
    (d / "calibration_provenance.json").write_text(json.dumps(prov, indent=1))
    log(f"<<< {arm} OK -> {d}")
    return prov


# ── verification of a finished directory, no GPU ──────────────────────────
def verify(out_root: Path) -> int:
    from export.stage_adapters import InspectQuantExportStage, StageContext

    out_root = out_root.resolve()  # see cmd_run: relative paths get doubled
    arms = sorted(p for p in out_root.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))
    if not arms:
        print(f"no arms under {out_root}")
        return 3
    bad = 0
    fps: dict[str, list[str]] = {}
    for d in arms:
        prov_p = d / "calibration_provenance.json"
        prov = json.loads(prov_p.read_text()) if prov_p.exists() else None
        # Absence must select the STRICTER gate, not the weaker one. First
        # version read residue-ness off the provenance alone, so an arm with no
        # provenance was treated as plain and skipped the widening check --
        # i.e. the arm we know least about got the least scrutiny. Caught by
        # running verify over a directory containing the broken export, which
        # passed. Fall back to the artefact on disk.
        if prov is not None:
            residue = prov.get("residue_ratio") is not None
        else:
            residue = (d / "residue_kernel_metadata.json").exists()
        params: dict[str, Any] = {
            "export_dir": str(d), "require_finite_export_tensors": True,
            "forbid_salient_indices_in_safetensors": True}
        if residue:
            params["expected_residue_metadata_min_layers"] = 1
            params["expected_extended_linear_dims_min"] = 1
        try:
            InspectQuantExportStage(f"verify_{d.name}")(StageContext(case_dir=d), params)
            status = "PASS"
        except Exception as exc:  # noqa: BLE001
            status = f"FAIL {type(exc).__name__}: {str(exc)[:110]}"
            bad += 1
        fp = (prov or {}).get("recipe_fingerprint", "NO-PROVENANCE")
        fps.setdefault(fp, []).append(d.name)
        print(f"  {d.name:<18} ratio={str((prov or {}).get('residue_ratio')):<6} "
              f"fp={fp[:8]:<8} {status}")
        if prov is None:
            bad += 1

    print()
    if set(fps) == {"NO-PROVENANCE"}:
        print("!! NO ARM CARRIES PROVENANCE -- which calibration produced these "
              "is unknown, so they cannot be asserted to be one table")
    elif len(fps) > 1:
        
        print("!! ARMS SPAN MORE THAN ONE CALIBRATION RECIPE -- not one table:")
        for fp, names in fps.items():
            print(f"     {fp[:8]}: {', '.join(names)}")
        bad += 1
    else:
        print(f"one recipe across all arms: {next(iter(fps))[:8]}")
    print("VERIFY " + ("PASS" if not bad else f"FAIL ({bad})"))
    return 0 if not bad else 4


def cmd_run(a) -> int:
    # Absolute, always: InspectQuantExportStage resolves `export_dir` relative
    # to the context's case_dir, so a relative --out silently becomes
    # <case_dir>/<case_dir> and the gate reports "No safetensors found" -- a
    # missing-file error for a file that is present. Found while testing the
    # gate, which is the argument for testing gates.
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = SPECS[a.spec]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    manifest: list[dict] = []
    for spec in arms:
        if spec.get("policy", "unset") is None and spec["arm"] == "anchor":
            log(f"skip {spec['arm']}: no policy given ({spec.get('note','')})")
            continue
        d = out / spec["arm"]
        if (d / "calibration_provenance.json").exists() and not a.force:
            log(f"skip {spec['arm']}: exists (use --force to rebuild)")
            manifest.append(json.loads((d / "calibration_provenance.json").read_text()))
            continue
        if d.exists():
            shutil.rmtree(d)
        manifest.append(build_arm(spec, out, gpu))
    
    
    
    
    
    
    
    
    
    
    
    for link, target in (("base_gptq_bf16", BASE_MODEL),
                         ("base_orig_bf16", ORIG_BF16)):
        dst = out / link
        if Path(target).is_dir():
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(target)
            log(f"reference link {link} -> {target}")
        else:
            log(f"WARNING reference {link}: target missing, {target}")

    (out / "manifest.json").write_text(json.dumps(
        {"spec": a.spec, "recipe_fingerprint": recipe_fingerprint(),
         "recipe_components": recipe_components(),
         "arms": manifest}, indent=1))
    log(f"manifest -> {out / 'manifest.json'}")
    return verify(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("preflight"); q.add_argument("--spec", default="p1_p4_4b")
    r = sub.add_parser("run")
    r.add_argument("--spec", default="p1_p4_4b"); r.add_argument("--out", required=True)
    r.add_argument("--force", action="store_true")
    v = sub.add_parser("verify"); v.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "preflight":
        return preflight(a.spec)
    if a.cmd == "verify":
        return verify(Path(a.out))
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
