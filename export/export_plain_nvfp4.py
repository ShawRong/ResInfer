# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib
import shutil

import torch


def main() -> int:
    # argparse, not raw sys.argv: this is the first command in the README's
    # Path A, and `--help` used to exit 1 with IndexError on sys.argv[2] --
    # so did running it with no arguments. The positional interface is
    # unchanged; only the two ways of asking it what it wants now work.
    ap = argparse.ArgumentParser(
        description="Legacy synthetic text-only NVFP4 smoke. For quality exports, "
                    "use python -m export.export_nvfp4 with an explicit dataset.")
    ap.add_argument("base", type=pathlib.Path,
                    help="bf16 base model: a local path, or a HF repo id "
                         "(which downloads)")
    ap.add_argument("out", type=pathlib.Path,
                    help="output directory for the exported checkpoint")
    a = ap.parse_args()
    base, out = a.base, a.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite export: {out}")

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(str(base))
    if any(getattr(config, key, None) is not None for key in ("vision_config", "audio_config")):
        raise ValueError("The legacy smoke supports text-only models. Use "
                         "python -m export.export_nvfp4 to preserve multimodal scopes.")
    out.mkdir(parents=True, exist_ok=True)

    print(f">>> loading base {base}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(base))
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # Wiring smoke only. These short prompts are not quality calibration.
    prompts = [
        "The capital of France is Paris, a city known for",
        "In machine learning, a transformer is a neural network that",
        "def fibonacci(n):\n    if n < 2:\n        return n",
        "The three primary colors are red, green, and blue, which",
        "Once upon a time, in a distant kingdom, there lived a",
        "The speed of light in a vacuum is approximately 299,792",
        "To make a cup of tea, first boil some water and then",
        "The mitochondria is often called the powerhouse of the cell",
    ] * 4

    def calib(m):
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt").input_ids.cuda()
                m(ids)

    print(">>> mtq.quantize NVFP4_DEFAULT_CFG (calibrating) ...", flush=True)
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=calib)

    print(f">>> export_hf_checkpoint -> {out}", flush=True)
    export_hf_checkpoint(model, export_dir=str(out))

    # Tokenizer files aren't emitted by the exporter; copy them so vLLM loads.
    for f in base.glob("*"):
        if f.suffix in {".json", ".model"} and not (out / f.name).exists():
            # Skip config.json (exporter writes its own) and any weight index
            # from a SHARDED base -- the exporter emits a single safetensors, so
            # a copied shard index would point at files that do not exist.
            if f.name in {"config.json"} or "index" in f.name:
                continue
            shutil.copy2(f, out / f.name)
    tok.save_pretrained(str(out))

    print(">>> exported files:", flush=True)
    for f in sorted(out.iterdir()):
        print(f"    {f.name:40s} {f.stat().st_size/1e6:8.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
