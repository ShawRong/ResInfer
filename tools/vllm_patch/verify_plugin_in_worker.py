# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Free local smoke: prove the `vllm.general_plugins` plugin (mechanism b)'
import os

os.environ["RESIDUE_PATCH_VLLM"] = "1"
os.environ["RESIDUE_PATCH_MODE"] = "mext_r1"
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get("RESIDUE_TEST_MODEL",
                       "Qwen/Qwen3-0.6B")


def main() -> None:
    print(f"[test] MAIN pid={os.getpid()}", flush=True)
    print(f"[test] VLLM_ENABLE_V1_MULTIPROCESSING="
          f"{os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '(unset=>1, subprocess)')}",
          flush=True)

    # NOTE: intentionally NOT calling patch_vllm_modelopt() here. The plugin is
    # the only thing that can install the patch in the EngineCore subprocess.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.35,
        enforce_eager=True,  # fast; multiprocessing still spawns EngineCore
    )
    out = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=8))
    print(f"[test] GEN: {out[0].outputs[0].text!r}", flush=True)
    print("[test] DONE — look for '(EngineCore pid=X) [residue:plugin] applied "
          "in pid=X' above with X != MAIN pid", flush=True)


if __name__ == "__main__":
    main()
