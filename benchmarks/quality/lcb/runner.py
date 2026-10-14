"""LiveCodeBench: native vLLM generation and pinned official scoring.

Use generate/run with a reviewed manifest and staged release_v6 data. score
consumes saved generations only; it never imports vLLM or initializes a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmarks.quality.common.results import input_hash
from benchmarks.quality.lcb.official import (
    COMMIT,
    PROTOCOL,
    checked_key,
    load_official,
    question_key,
    score_code,
)
from benchmarks.quality.common.protocol import load_manifest, resolve_arms, SAMPLING_FIELDS


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def read_questions(path, manifest, limit=None):
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        rows = json.loads(path.read_text())
    if not rows or len({question_key(row) for row in rows}) != len(rows):
        raise ValueError("Empty dataset or duplicate LCB question IDs")
    if len(rows) != manifest["n_prompts"]:
        raise ValueError("Dataset count differs from the frozen manifest")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--max-samples must be positive")
        rows = rows[:limit]
    names = {f.name for f in fields(CodeGenerationProblem)}
    samples = []
    for row in rows:
        problem = CodeGenerationProblem(**{k: row[k] for k in names})
        sample = problem.get_evaluation_sample()
        samples.append(
            {
                "item_id": question_key(row),
                "problem": row,
                "sample": sample,
                "sample_sha256": input_hash(sample),
                "checker": checked_key(row, manifest["release_version"], sample),
            }
        )
    return samples


def runtime_args(arm, manifest, args, out):
    return SimpleNamespace(
        model=arm["abs_path"],
        arm=arm["id"],
        out=str(out),
        repo=str(Path(__file__).resolve().parents[3]),
        bf16=arm["kind"] == "fp16",
        plain=arm["kind"] != "residue",
        quantization=None if arm["kind"] == "fp16" else "modelopt_fp4",
        mext_max_m=args.mext_max_m,
        residue_off=arm.get("mext") == "off",
        prefill_residue_off=False,
        prefill_only=None,
        seed=manifest["seed"],
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_mem_util=args.gpu_mem_util,
        flashinfer_autotune=args.flashinfer_autotune,
        engine_config=args.engine_config,
    )


def sampling_kwargs(manifest):
    params = {k: manifest[k] for k in SAMPLING_FIELDS if k != "max_new_tokens"}
    params.update(
        max_tokens=manifest["max_new_tokens"],
        n=manifest["num_samples"],
        skip_special_tokens=False,
    )
    return params


def final_answer(text, token_ids, tokenizer, thinking):
    if not thinking:
        return text, None
    boundary = tokenizer.encode("</think>", add_special_tokens=False)
    if len(boundary) != 1:
        raise ValueError(
            "This manifest requires a tokenizer with a single </think> token"
        )
    if boundary[0] not in token_ids:
        return "", False
    end = token_ids.index(boundary[0])
    return tokenizer.decode(token_ids[end + 1 :], skip_special_tokens=True), True


def generate_arm(args, manifest, arm):
    from benchmarks.quality.common.runtime import (
        install_env,
        pinned_engine_kwargs,
        write_effective_engine_config,
        write_provenance,
    )

    out = Path(args.out) / arm["id"]
    out.mkdir(parents=True, exist_ok=False)
    runtime = runtime_args(arm, manifest, args, out)
    for key in list(os.environ):
        if key.startswith("RESIDUE_"):
            os.environ.pop(key)
    install_env(runtime)
    from vllm import LLM, SamplingParams
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.lm_styles import LMStyle

    samples = read_questions(args.data, manifest, args.max_samples)
    engine = pinned_engine_kwargs(runtime, manifest["max_model_len"])
    engine["generation_config"] = "vllm"
    params = SamplingParams(**sampling_kwargs(manifest))
    batch_size = engine["max_num_seqs"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("max_num_seqs must be a positive integer")
    llm = LLM(
        model=runtime.model,
        tokenizer=runtime.model,
        dtype="bfloat16",
        quantization=runtime.quantization,
        trust_remote_code=True,
        **engine,
    )
    tokenizer = llm.get_tokenizer()
    template_kwargs = {"enable_thinking": manifest["enable_thinking"]}
    if "reasoning_effort" in manifest:
        template_kwargs["reasoning_effort"] = manifest["reasoning_effort"]
    names = {f.name for f in fields(CodeGenerationProblem)}
    for sample in samples:
        problem = CodeGenerationProblem(**{k: sample["problem"][k] for k in names})
        messages = format_prompt_generation(problem, LMStyle.OpenAIChat)
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if len(ids) + manifest["max_new_tokens"] > engine["max_model_len"]:
            raise ValueError(
                f"{sample['item_id']}: prompt plus output budget exceeds context"
            )
        sample.update(
            prompt_token_ids=ids,
            prompt=tokenizer.decode(ids, skip_special_tokens=False),
        )
    # The model-specific protocol is frozen before the first request.
    write_json(out / "samples.json", samples)
    write_json(
        out / "run.json",
        {
            "protocol": PROTOCOL,
            "official_commit": COMMIT,
            "manifest": manifest,
            "arm": arm,
            "sampling": sampling_kwargs(manifest),
            "template_kwargs": template_kwargs,
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "samples_sha256": input_hash(samples),
            "submission_batch_size": batch_size,
            "tokenizer_sha256": input_hash(
                {
                    "backend": tokenizer.backend_tokenizer.to_str(),
                    "chat_template": tokenizer.chat_template,
                    "special_tokens": tokenizer.special_tokens_map,
                }
            ),
            "model_metadata_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(runtime.model).glob("*.json")
            },
            "adapter_sha256": {
                str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parents[1].rglob("*")
                if p.is_file() and p.suffix in (".py", ".json")
            },
        },
    )
    write_effective_engine_config(llm, runtime)
    write_provenance(runtime, out, engine, {"venue": "lcb", "protocol": PROTOCOL})
    start = time.monotonic()
    with (out / "generations.jsonl").open("x") as stream:
        for offset in range(0, len(samples), batch_size):
            batch = samples[offset : offset + batch_size]
            began = time.monotonic()
            outputs = llm.generate(
                [{"prompt_token_ids": s["prompt_token_ids"]} for s in batch],
                params,
                use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise RuntimeError("Incomplete vLLM batch")
            for sample, output in zip(batch, outputs):
                if (
                    list(output.prompt_token_ids) != sample["prompt_token_ids"]
                    or len(output.outputs) != manifest["num_samples"]
                ):
                    raise RuntimeError("Mismatched prompt or incomplete sampling")
                for repeat, completion in enumerate(output.outputs):
                    tokens = list(completion.token_ids)
                    if not tokens or completion.finish_reason not in ("stop", "length"):
                        raise RuntimeError("Empty or aborted LCB completion")
                    answer, closed = final_answer(
                        completion.text, tokens, tokenizer, manifest["enable_thinking"]
                    )
                    record = {
                        "item_id": sample["item_id"],
                        "repeat": repeat,
                        "sample_sha256": sample["sample_sha256"],
                        "prompt_sha256": input_hash(sample["prompt_token_ids"]),
                        "text": completion.text,
                        "token_ids": tokens,
                        "answer_text": answer,
                        "thinking_closed": closed,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "batch_seconds": time.monotonic() - began,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
    write_json(
        out / "generated.json",
        {
            "seconds": time.monotonic() - start,
            "generations_sha256": hashlib.sha256(
                (out / "generations.jsonl").read_bytes()
            ).hexdigest(),
        },
    )


def checked_generations(directory):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text())
    samples = json.loads((directory / "samples.json").read_text())
    if run["protocol"] != PROTOCOL or input_hash(samples) != run["samples_sha256"]:
        raise ValueError("Changed LCB protocol or frozen samples")
    rows = [
        json.loads(line)
        for line in (directory / "generations.jsonl").read_text().splitlines()
    ]
    expected = {
        (s["item_id"], i)
        for s in samples
        for i in range(run["manifest"]["num_samples"])
    }
    actual = [(r["item_id"], r["repeat"]) for r in rows]
    if not expected or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected LCB generations")
    by_id = {s["item_id"]: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate frozen LCB question IDs")
    for sample in samples:
        if input_hash(sample["sample"]) != sample["sample_sha256"]:
            raise ValueError("Changed LCB test content")
    for row in rows:
        sample = by_id[row["item_id"]]
        if row["sample_sha256"] != sample["sample_sha256"] or row[
            "prompt_sha256"
        ] != input_hash(sample["prompt_token_ids"]):
            raise ValueError("Generation belongs to another LCB input")
        for key in (
            "text",
            "answer_text",
            "token_ids",
            "finish_reason",
            "stop_reason",
            "thinking_closed",
        ):
            if key not in row:
                raise ValueError(f"Incomplete generation record: {key}")
        if not row["token_ids"] or row["finish_reason"] not in ("stop", "length"):
            raise ValueError("Empty or aborted LCB completion")
    done = json.loads((directory / "generated.json").read_text())
    if (
        done["generations_sha256"]
        != hashlib.sha256((directory / "generations.jsonl").read_bytes()).hexdigest()
    ):
        raise ValueError("Generation file changed after completion")
    return run, samples, {(r["item_id"], r["repeat"]): r for r in rows}


def score(args):
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results

    run, samples, generations = checked_generations(args.generations)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    with (out / "scores.jsonl").open("x") as stream:
        for index, sample in enumerate(samples):
            key = checked_key(
                sample["problem"], run["manifest"]["release_version"], sample["sample"]
            )
            results[index] = []
            for repeat in range(run["manifest"]["num_samples"]):
                response = generations[sample["item_id"], repeat]
                code = extract_code(response["answer_text"], LMStyle.OpenAIChat)
                verdict = score_code(
                    args.lcb_source, sample["sample"], code, key, timeout=6
                )
                # Upstream pass@k treats an empty test list as an all-pass.
                results[index].append(verdict["tests"] or [-2])
                stream.write(
                    json.dumps(
                        {
                            **verdict,
                            "item_id": sample["item_id"],
                            "repeat": repeat,
                            "response_sha256": input_hash(response),
                            "code": code,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
    metrics = compute_metrics_from_results(results, k_list=[1])
    write_json(
        out / "summary.json",
        {
            "protocol": PROTOCOL,
            "official_commit": COMMIT,
            "metrics": metrics,
            "questions": len(samples),
            "num_samples": run["manifest"]["num_samples"],
            "samples_sha256": run["samples_sha256"],
            "status": "complete",
            "length_stops": sum(
                r["finish_reason"] == "length" for r in generations.values()
            ),
            "missing_thinking_end": sum(
                r["thinking_closed"] is False for r in generations.values()
            ),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "score", "run"))
    parser.add_argument(
        "--lcb-source",
        required=True,
        help="fixed external checkout with reviewed patch",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--models-root", default=".")
    parser.add_argument("--data", help="frozen official dataset JSON/JSONL/parquet")
    parser.add_argument("--arms")
    parser.add_argument(
        "--generations", help="completed generation directory (score mode)"
    )
    parser.add_argument("--scorer-python", default=sys.executable)
    parser.add_argument("--engine-config")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument(
        "--flashinfer-autotune", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mext-max-m", type=int, default=64)
    args = parser.parse_args(argv)
    load_official(args.lcb_source)
    if args.mode == "score":
        if not args.generations:
            parser.error("score requires --generations")
        score(args)
        return 0
    if not args.manifest or not args.data or args.max_num_seqs <= 0:
        parser.error(
            "generate/run requires --manifest, --data and a positive batch size"
        )
    manifest = load_manifest(args.manifest)
    if (
        manifest.get("benchmark") != "lcb"
        or manifest.get("release_version") != "release_v6"
    ):
        raise ValueError(
            "Use the explicit official LCB release_v6 manifest; do not reinterpret old task names"
        )
    for name in ("num_samples", "n_prompts", "max_new_tokens", "max_model_len"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"Manifest requires a positive explicit {name}")
    for name in SAMPLING_FIELDS:
        if not isinstance(manifest[name], (int, float)) or not math.isfinite(
            manifest[name]
        ):
            raise ValueError(f"Nonfinite or invalid sampling parameter: {name}")
    arms = resolve_arms(manifest, args.models_root, args.arms)
    original = list(sys.argv[1:] if argv is None else argv)
    if args.mode == "run":
        # Generation exits before CPU scoring, releasing GPU/model memory.
        for arm in arms:
            child = ["generate", *original[1:]]
            # Argparse uses the last occurrence, including --arms=... syntax.
            child += ["--arms", arm["id"]]
            subprocess.run(
                [sys.executable, "-m", "benchmarks.quality.lcb", *child], check=True
            )
            subprocess.run(
                [
                    args.scorer_python,
                    "-m",
                    "benchmarks.quality.lcb",
                    "score",
                    "--lcb-source",
                    args.lcb_source,
                    "--generations",
                    str(Path(args.out) / arm["id"]),
                    "--out",
                    str(Path(args.out) / (arm["id"] + "-scores")),
                ],
                check=True,
            )
    else:
        if len(arms) != 1:
            raise ValueError(
                "generate requires exactly one --arms selection; use run for several arms"
            )
        generate_arm(args, manifest, arms[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
