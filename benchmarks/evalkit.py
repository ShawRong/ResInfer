'ResInfer evaluation kit -- one self-contained file, four venues, one gate.'

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import statistics as st
import string
import sys
from collections import Counter
from difflib import SequenceMatcher

# Official LongBench v1 prompts + generation caps, inlined so the kit needs no
# config files. Verbatim from THUDM/LongBench config/dataset2prompt.json and
# dataset2maxlen.json.
PROMPTS = {
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
        "following is an abstract.\n\n{input}\n\nPlease enter the number of the "
        "paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"),
    "musique": None,      # same template as hotpotqa
    "2wikimqa": None,     # same template as hotpotqa
}
PROMPTS["musique"] = PROMPTS["2wikimqa"] = PROMPTS["hotpotqa"]
MAXGEN = {"passage_retrieval_en": 32, "hotpotqa": 32, "musique": 32, "2wikimqa": 32}
METRIC = {"passage_retrieval_en": "retrieval", "hotpotqa": "qa_f1",
          "musique": "qa_f1", "2wikimqa": "qa_f1"}
# Official: these subsets are raw completions, not chat turns. None of the four
# above is in that set, but keep the rule explicit so adding one is safe.
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
DEFAULT_TASKS = "passage_retrieval_en,2wikimqa,musique,hotpotqa"


# ----------------------------------------------------------------- metrics
def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gt: str) -> float:
    p, g = _norm(pred).split(), _norm(gt).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def qa_f1(pred, gts):
    return max(_f1(pred, g) for g in gts)


def retrieval(pred, gts):
    m = re.findall(r"Paragraph (\d+)", gts[0])
    if not m:
        return 0.0
    nums = re.findall(r"\d+", pred)
    return sum(1 for n in nums if n == m[0]) / len(nums) if nums else 0.0


def code_sim(pred, gts):
    line = ""
    for cand in pred.lstrip("\n").split("\n"):
        if "`" not in cand and "#" not in cand and "//" not in cand:
            line = cand
            break
    return SequenceMatcher(None, line, gts[0]).ratio()


SCORERS = {"qa_f1": qa_f1, "retrieval": retrieval, "code_sim": code_sim}


# ------------------------------------------------- downstream task registry
# Added rather than reaching for lm-eval: lm_eval 0.4.9.2 does not import under
# transformers 5.12.1 (AutoModelForVision2Seq was renamed), and patching one
# name only surfaces the next. Building the tasks here keeps every number in
# the paper on ONE code path -- same determinism contract, same per-item jsonl,
# same paired-bootstrap analyze as the LongBench arms already validated.
#
# Two kinds:
#   gen  -- generate, then exact-match the extracted answer (GSM8K)
#   mc   -- score each choice by summed continuation logprob, take argmax
#           (ARC / HellaSwag / PIQA / WinoGrande). This reuses the
#           prompt_logprobs path built for `ppl`, so it needs no new plumbing.

def _gsm8k_ans(text: str):
    """Last number in the string, upstream's convention."""
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def gsm8k_score(pred: str, gts) -> float:
    # Upstream marks the gold answer after '####'.
    gold = gts[0].split("####")[-1].strip().replace(",", "")
    got = _gsm8k_ans(pred.split("Question:")[0])
    try:
        return 1.0 if got is not None and abs(float(got) - float(gold)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if got == gold else 0.0


SCORERS["gsm8k"] = gsm8k_score

# Where each generative task's scorer stops reading. Kept next to the scorer so
# the two cannot drift: a stop sequence that cuts BEFORE the scored span turns
# every answer into a miss, and one that cuts after it only wastes tokens.
STOPS = {"gsm8k": ["Question:"]}

# name -> (hf path, config, split, kind)
DOWNSTREAM = {
    "gsm8k":         ("gsm8k", "main", "test", "gen"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", "mc"),
    "hellaswag":     ("Rowan/hellaswag", None, "validation", "mc"),
    "piqa":          ("ybisk/piqa", None, "validation", "mc"),
    "winogrande":    ("allenai/winogrande", "winogrande_xl", "validation", "mc"),
}


def normalise_downstream(name: str, row: dict) -> dict:
    """One shape for every task: {context, choices, gold} or {context, answers}.

    Each upstream dataset spells its fields differently; do the translation
    once, here, so the scoring loop never branches on task name.
    """
    if name == "gsm8k":
        return {"context": "Question: " + row["question"] + "\nAnswer:",
                "answers": [row["answer"]]}
    if name == "arc_challenge":
        ch = row["choices"]
        gold = ch["label"].index(row["answerKey"]) if row["answerKey"] in ch["label"] else 0
        return {"context": "Question: " + row["question"] + "\nAnswer:",
                "choices": [" " + c for c in ch["text"]], "gold": gold}
    if name == "hellaswag":
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize() if row.get("ctx_b") else row["ctx"]
        return {"context": row["activity_label"] + ": " + ctx,
                "choices": [" " + e for e in row["endings"]], "gold": int(row["label"])}
    if name == "piqa":
        return {"context": "Question: " + row["goal"] + "\nAnswer:",
                "choices": [" " + row["sol1"], " " + row["sol2"]], "gold": int(row["label"])}
    if name == "winogrande":
        # Upstream scores the SHARED suffix after the blank, with each option
        # substituted in -- comparing equal-length continuations. Scoring the
        # option words themselves would compare different token counts and
        # favour whichever happens to be shorter.
        idx = row["sentence"].index("_")
        suffix = row["sentence"][idx + 1:]
        return {"context": "", "gold": int(row["answer"]) - 1,
                "choices": [row["sentence"][:idx] + row[f"option{i}"] + "\u0000" + suffix
                            for i in (1, 2)]}
    raise KeyError(name)


def bootstrap_se(diffs, n_boot=4000, seed=0):
    r = random.Random(seed)
    n = len(diffs)
    return st.pstdev([sum(diffs[r.randrange(n)] for _ in range(n)) / n
                      for _ in range(n_boot)])


# ------------------------------------------------------------------- fetch
def cmd_fetch(a) -> int:
    """Materialise each subset to a local jsonl, once. Offline-friendly."""
    out = pathlib.Path(a.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for t in a.tasks.split(","):
        t = t.strip()
        f = out / f"{t}.jsonl"
        if f.exists() and f.stat().st_size > 0:
            print(f"  {t}: already local ({sum(1 for _ in f.open())} rows)")
            continue
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", t, split="test")
            with f.open("w") as fh:
                for x in ds:
                    fh.write(json.dumps({k: x[k] for k in
                                         ("context", "input", "answers")},
                                        ensure_ascii=False) + "\n")
            print(f"  {t}: fetched {len(ds)} rows -> {f}")
        except Exception as e:
            missing.append(t)
            print(f"  {t}: FAILED {type(e).__name__}: {str(e)[:140]}")
    if missing:
        print("\nRELAY REQUEST -- these subsets could not be fetched on the pod.")
        print("Each is a jsonl of {context, input, answers} from")
        print("  load_dataset('THUDM/LongBench', <subset>, split='test')")
        print("Total for all four is ~110 MB. Place at:")
        for t in missing:
            print(f"  {out}/{t}.jsonl")
        return 3
    return 0


# --------------------------------------------------------------------- run
def install_env(a):
    """Every RESIDUE_* the workers need, set BEFORE vllm is imported."""
    # Resolve ONLY a real local path. Calling .resolve() unconditionally turns
    # a Hugging Face repo id ("Qwen/Qwen3-4B") into "<cwd>/Qwen/Qwen3-4B", and
    # the failure surfaces as HFValidationError about a malformed repo id --
    # which reads like the caller typed it wrong, not like we mangled it.
    if pathlib.Path(a.model).exists():
        a.model = str(pathlib.Path(a.model).resolve())
    if a.bf16:
        a.plain = True
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if not a.plain:
        os.environ["RESIDUE_NVFP4"] = "1"
        os.environ["RESIDUE_VLLM_MODEL_DIR"] = a.model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        os.environ["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
        os.environ["RESIDUE_FORCE_DECODE_MEXT"] = "0"
        os.environ["RESIDUE_DECODE_MEXT_MAX_M"] = str(getattr(a, "mext_max_m", 64))
        # Cubins force modelopt_fp4 onto the trtllm-gen path, which bypasses or
        # breaks the residue patch (two observed failure modes). They belong to
        # non-residue trtllm A/B arms only. Drop them rather than trust the
        # caller's environment.
        for var in ("FLASHINFER_CUBIN_DIR", "VLLM_HAS_FLASHINFER_CUBIN"):
            if os.environ.pop(var, None) is not None:
                print(f"[kit] dropped {var}: incompatible with a residue arm")
    if getattr(a, "decode_mext_off", False):
        raise SystemExit(
            "--decode-mext-off is not implementable today. RESIDUE_ENABLE_DECODE_MEXT=0\n"
            "breaks weight loading (KeyError 'weight_scale_base', vLLM 0.26) and\n"
            "RESIDUE_DECODE_MEXT_MAX_M=0 silently keeps residue on. See install_env().")
    if getattr(a, "prefill_residue_off", False) or getattr(a, "residue_off", False):
        if a.plain:
            raise SystemExit("--prefill-residue-off is meaningless with --plain/--bf16")
        os.environ["RESIDUE_PREFILL_RESIDUE_OFF"] = "1"
        if getattr(a, "residue_off", False):
            # threshold 0 -> every M is "> 0" -> residue killed everywhere
            os.environ["RESIDUE_PREFILL_RESIDUE_OFF_MIN_M"] = "0"
    # The fourth cell of the kill-switch 2x2, and the only one no eval arm had
    # ever used: MAX_M kills residue in the DECODE band, leaving prefill-only.
    # Without it `both ~= prefill + decode` (complementary) and
    # `both ~= max(prefill, decode)` (redundant) are indistinguishable -- the
    # source comment in residue_linear_op.py says so, and 24 arms on disk had
    # gone by with the cell unused because reaching it needed a raw env var.
    if getattr(a, "prefill_only", None) is not None:
        if a.plain:
            raise SystemExit("--prefill-only is meaningless with --plain/--bf16")
        os.environ["RESIDUE_PREFILL_RESIDUE_OFF"] = "1"
        os.environ["RESIDUE_PREFILL_RESIDUE_OFF_MAX_M"] = str(a.prefill_only)
        if os.environ.get("RESIDUE_PREFILL_RESIDUE_OFF_MIN_M") is not None:
            # Both bounds set means "kill where either holds", i.e. residue
            # survives only between them -- not what anyone reaching for
            # --prefill-only wants, and it would silently produce the plain arm
            # under the prefill-only name.
            raise SystemExit(
                "--prefill-only cannot be combined with --residue-off: both "
                "bounds set kills BOTH bands and yields the plain cell.")
    repo = a.repo or os.environ.get("RESINFER_REPO", "")
    if repo:
        parts = [repo, str(pathlib.Path(repo) / "csrc")]
        os.environ["PYTHONPATH"] = ":".join(
            [p for p in parts + [os.environ.get("PYTHONPATH", "")] if p])
        # spawn workers read PYTHONPATH, but THIS process fixed sys.path at
        # startup -- the vLLM plugin that teaches the loader the extended-K
        # weight shapes must import `quant` HERE too, or loading dies on
        # `assert param_data.shape == loaded_weight.shape`.
        for p in reversed(parts):
            if p not in sys.path:
                sys.path.insert(0, p)
    
    
    
    
    
    
    
    
    
    
    
    if os.environ.get("RESIDUE_EVALKIT_ALLOW_CACHE", "0") == "1":
        
        
        
        
        
        
        
        print("[kit] " + "!" * 62, flush=True)
        print("[kit] !! RESIDUE_EVALKIT_ALLOW_CACHE=1 -- torch.compile cache is ON.")
        print("[kit] !! NUMBERS FROM THIS RUN ARE NOT PUBLISHABLE. An arm's score")
        print("[kit] !! depends on what ran before it in the queue.")
        print("[kit] " + "!" * 62, flush=True)
    else:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    # Record the execution contract in the run's own output. Two runs that
    # differ in any of these are not comparable, and today that difference is
    # invisible after the fact -- the scores carry no trace of it.
    print("[kit] execution contract: " + " ".join(
        f"{k}={os.environ.get(k, '<unset>')}" for k in (
            "VLLM_ENABLE_V1_MULTIPROCESSING", "RESIDUE_NVFP4",
            "RESIDUE_ENABLE_DECODE_MEXT", "RESIDUE_PREFILL_RESIDUE_OFF",
            "RESIDUE_PREFILL_RESIDUE_OFF_MIN_M",
            "RESIDUE_PREFILL_RESIDUE_OFF_MAX_M",
            "VLLM_DISABLE_COMPILE_CACHE")), flush=True)
    # flashinfer JIT links -lcuda from a stub dir that may not contain it
    os.environ["LIBRARY_PATH"] = ("/usr/lib/x86_64-linux-gnu:"
                                  + os.environ.get("LIBRARY_PATH", ""))


def cmd_run(a) -> int:
    install_env(a)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config.kernel import KernelConfig

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    todo = [t for t in tasks
            if not (out_dir / f"{a.arm}__{t}.jsonl").exists() or a.force]
    if not todo:
        print(f"[kit] {a.arm}: all {len(tasks)} subsets already on disk; nothing to do")
        return 0
    print(f"[kit] arm={a.arm} subsets={todo}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    
    
    ekw = pinned_engine_kwargs(a, a.max_model_len)
    write_provenance(a, out_dir, ekw, {"venue": "longbench_v1", "subsets": todo,
                                      "max_gen": {t: MAXGEN[t] for t in todo}})
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              quantization=None if a.bf16 else "modelopt_fp4", **ekw)

    for task in todo:
        f = pathlib.Path(a.data_dir) / f"{task}.jsonl"
        if f.exists():
            items = [json.loads(l) for l in f.open() if l.strip()]
        else:
            from datasets import load_dataset
            items = [dict(x) for x in load_dataset("THUDM/LongBench", task,
                                                   split="test")]
        if a.limit:
            items = items[: a.limit]
        tpl, max_gen = PROMPTS[task], MAXGEN[task]
        budget = a.max_model_len - max_gen - 64
        texts = []
        for it in items:
            raw = tpl.format(context=it["context"], input=it["input"])
            ids = tok(raw, add_special_tokens=False)["input_ids"]
            if len(ids) > budget:            # official: keep head and tail
                half = budget // 2
                raw = (tok.decode(ids[:half], skip_special_tokens=True)
                       + tok.decode(ids[-half:], skip_special_tokens=True))
            if task not in NO_CHAT:
                raw = tok.apply_chat_template(
                    [{"role": "user", "content": raw}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            texts.append(raw)

        outs = llm.generate(texts, SamplingParams(temperature=0.0,
                                                  max_tokens=max_gen,
                                                  seed=a.seed))
        scorer = SCORERS[METRIC[task]]
        rows, ptok, gtok = [], 0, 0
        for it, o in zip(items, outs):
            pred = o.outputs[0].text.strip()
            rows.append({"score": scorer(pred, it["answers"]),
                         "pred": pred[:300], "answers": it["answers"],
                         "n_prompt": len(o.prompt_token_ids),
                         "n_gen": len(o.outputs[0].token_ids)})
            ptok += len(o.prompt_token_ids)
            gtok += len(o.outputs[0].token_ids)
        (out_dir / f"{a.arm}__{task}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        score = sum(r["score"] for r in rows) / len(rows)
        print(f"[kit] {a.arm:<20} {task:<22}{METRIC[task]:<11}{score:.4f}  "
              f"n={len(rows)}  prefill={ptok/max(ptok+gtok,1)*100:.1f}%  "
              f"prompt_med={sorted(r['n_prompt'] for r in rows)[len(rows)//2]:,}",
              flush=True)
    print(f"[kit] wrote {out_dir}/{a.arm}__*.jsonl")
    return 0


# ----------------------------------------------------------------- analyze
def cmd_analyze(a) -> int:
    """Reads ONLY the jsonl on disk. No GPU, seconds, re-runnable forever."""
    d = pathlib.Path(a.out)
    def _prov(arm_id):
        p = pathlib.Path(a.out) / f"{arm_id}__provenance.json"
        return json.loads(p.read_text()) if p.exists() else None

    arms, tasks = {}, set()
    for f in sorted(d.glob("*__*.jsonl")):
        arm, task = f.stem.split("__", 1)
        arms.setdefault(arm, {})[task] = [json.loads(l) for l in f.open() if l.strip()]
        tasks.add(task)
    if not arms:
        print(f"no results under {d}")
        return 1
    tasks = sorted(tasks)
    # Leftovers are APPENDED, not dropped. A stale --order silently hiding an
    # arm that ran is the same failure as a silently absent one: the table
    # looks complete. cmd_analyze_ppl already did this correctly.
    _req = [x.strip() for x in (a.order or "").split(",") if x.strip()]
    order = ([x for x in _req if x in arms]
             + [x for x in sorted(arms) if x not in _req]) or sorted(arms)

    print(f"\n=== scores ({d}) ===")
    print(f"{'subset':<22}" + "".join(f"{x:>14}" for x in order))
    for t in tasks:
        line = f"{t:<22}"
        for x in order:
            r = arms[x].get(t)
            line += f"{sum(i['score'] for i in r)/len(r):>14.4f}" if r else f"{'-':>14}"
        print(line)

    
    
    
    
    
    
    
    
    
    
    if any(i.get("n_prompt") is not None for x in order for i in arms[x].get(tasks[0], [])):
        print(f"\n=== prompt length (median tokens) ===")
        print(f"{'subset':<22}" + "".join(f"{x:>14}" for x in order))
        for t in tasks:
            line = f"{t:<22}"
            for x in order:
                r = arms[x].get(t)
                if not r or r[0].get("n_prompt") is None:
                    line += f"{'-':>14}"
                    continue
                line += f"{sorted(i['n_prompt'] for i in r)[len(r) // 2]:>14,}"
            print(line)

    
    
    
    
    
    
    
    
    
    
    
    if any(i.get("n_gen") is not None for x in order for i in arms[x].get(tasks[0], [])):
        print(f"\n=== generation length (median tokens; x/ref) ===")
        print(f"{'subset':<22}" + "".join(f"{x:>14}" for x in order))
        for t in tasks:
            line = f"{t:<22}"
            rr = arms.get(a.reference, {}).get(t)
            ref_med = (sorted(i["n_gen"] for i in rr)[len(rr) // 2]
                       if rr and rr[0].get("n_gen") is not None else None)
            for x in order:
                r = arms[x].get(t)
                if not r or r[0].get("n_gen") is None:
                    line += f"{'-':>14}"
                    continue
                med = sorted(i["n_gen"] for i in r)[len(r) // 2]
                line += (f"{med:>8,}{'':>6}" if not ref_med
                         else f"{med:>8,}{med / ref_med:>6.2f}")
            print(line)
        if all(m <= 64 for m in (sorted(i["n_gen"] for i in arms[x][t])[len(arms[x][t]) // 2]
                                 for x in order for t in tasks if arms[x].get(t))):
            print("  Note: every median is <=64 tokens, so this venue caps "
                  "generation and the column carries no information about how "
                  "much an arm reasons. Read it on a decode venue (LCB, AIME).")

        # Venue-qualification criterion 2, made mechanical.
        #
        # A venue can pass the damage test and still be scoring something other
        # than the capability it names. On passage_retrieval_en, wrong answers
        # sit at exactly the 32-token cap while right answers are 4-5 tokens,
        # and bf16 hits the cap 0 times against the quantised arms' 29-39: the
        # model paraphrases the passage instead of naming it and the cap turns
        # that into a zero. That subset carried the largest "damage" in this
        # project's tables for months. Criterion 1 cannot see it, so it gets
        # its own column rather than a footnote somebody has to recompute.
        print("\n=== truncation check (criterion 2: is the venue scoring the cap?) ===")
        print(f"  {'arm':<22}{'subset':<22}{'n_gen ok':>9}{'n_gen bad':>10}"
              f"{'bad@cap':>9}{'  verdict'}")
        flagged = False
        nostops: list[str] = []
        for x in order:
            for t in tasks:
                rows = arms[x].get(t) or []
                if not rows or rows[0].get("n_gen") is None:
                    continue
                ok = [i["n_gen"] for i in rows if i.get("score", 0) >= 1]
                bad = [i["n_gen"] for i in rows if i.get("score", 0) < 1]
                if not bad:
                    continue
                mo = sorted(ok)[len(ok) // 2] if ok else 0
                mb = sorted(bad)[len(bad) // 2]
                # Read the declared cap; only infer it when the run did not
                # record one. Inferring it from `max(n_gen)` is the same
                # inference-instead-of-reading this whole check exists to
                # catch: with no item near the real ceiling, the longest
                # generation masquerades as the cap.
                declared = ((_prov(x) or {}).get("max_gen") or {}).get(t)
                cap = int(declared) if declared else max(i["n_gen"] for i in rows)
                at_cap = sum(1 for v in bad if v >= cap)
                # Signature A: wrong answers pinned at the ceiling while right
                # answers are far below it -- the venue is scoring truncation.
                suspect = at_cap >= max(3, 0.5 * len(bad)) and mo and mb >= 2 * mo
                # Signature B, which A misses and which cost a false negative
                # the day this check shipped: EVERYTHING at the cap. On GSM8K
                # every generation ran to 512 because the prompt has no stop
                # condition, so mb == mo and A's ratio test cannot fire. That
                # is not automatically fatal -- if the scorer reads a prefix
                # (here: the text before the model's next "Question:"), the
                # score is unaffected and the cost is wasted tokens. If it
                # reads the whole generation, the cap is deciding scores.
                # Reported rather than asserted, because which one it is lives
                # in the scorer, not here.
                nostop = mo and mb and mo >= cap and mb >= cap
                if suspect:
                    flagged = True
                if nostop:
                    nostops.append(f"{x}/{t}")
                note = ("  <-- SCORING THE CAP" if suspect
                        else "  <-- NO STOP CONDITION" if nostop else "")
                print(f"  {x:<22}{t:<22}{mo:>9,}{mb:>10,}{at_cap:>9}{note}")
        if nostops and not flagged:
            print(f"\n  note: every generation ran to the cap for {len(nostops)} arm/subset")
            print("  pair(s) -- the prompt has no stop condition. Harmless IF the")
            print("  scorer reads a prefix of the generation; if it reads all of it,")
            print("  the cap is deciding the score. Check the scorer, then add a stop")
            print("  sequence: it makes the run cheaper and the scored span identical.")
        if flagged:
            print("\n  !! At least one arm's wrong answers are pinned at the")
            print("  !! generation cap while its right answers are far shorter.")
            print("  !! This venue is scoring instruction-following/truncation,")
            print("  !! not the capability it is named for. It FAILS criterion 2")
            print("  !! regardless of how large its measured damage is.")

    ref, base = a.reference, a.baseline

    
    
    
    
    
    
    
    
    
    
    
    def _cell(arm_id):
        """Which of the kill-switch 2x2 an arm actually ran, from provenance.

        Never from the arm's name or the flag's name: the gate is called
        PREFILL and with _MAX_M it kills the DECODE band. A whole session was
        spent reading "off" as "the method is off" -- it is not; without a
        bound, residue survives everywhere below the fold band, so the delta
        an "off" arm defines is the PREFILL half only.
        """
        rec = _prov(arm_id)
        if rec is None:
            return None, "no provenance.json -- the cell cannot be determined"
        env = rec.get("env") or {}
        # Historical provenance uses the old gate name. Prefer the current
        # name when present; this fallback only reads saved result records.
        nvfp4 = env.get("RESIDUE_NVFP4", env.get("RESIDUE_PATCH_VLLM"))
        if nvfp4 != "1":
            return "stock", "stock vLLM, residue patch never installed"
        if env.get("RESIDUE_PREFILL_RESIDUE_OFF") != "1":
            return "both", "residue active everywhere"
        lo = env.get("RESIDUE_PREFILL_RESIDUE_OFF_MIN_M")
        hi = env.get("RESIDUE_PREFILL_RESIDUE_OFF_MAX_M")
        if lo == "0" and hi is None:
            return "plain", "residue killed at every M (K still extended)"
        if hi is not None and lo is None:
            return "prefill-only", f"residue killed in the decode band (M <= {hi})"
        if hi is not None and lo is not None:
            return "between", ("BOTH bounds set -- residue survives only between "
                               "them; almost certainly not what was intended")
        return "decode-only", ("residue killed above the fold band -- PREFILL "
                               "half only; decode residue is still live")

    def _floor_kind(arm_id):
        cell, why = _cell(arm_id)
        if cell in (None, "stock"):
            return ("stock" if cell else None), why
        if cell == "both":
            return "method", "residue ACTIVE -- this is not a floor at all"
        if cell == "plain":
            return "runtime_off", why
        return cell, why

    kind, why = _floor_kind(base)
    if kind is not None:
        print(f"\n  floor: {base} -- {why}")
        if kind == "runtime_off":
            print("  This isolates the MECHANISM (same weights, one env var). It is")
            print('  NOT the deployment alternative. Compare separately calibrated')
            print('  stock plain exports when assessing deployment quality. Recovery')
            print('  relative to a runtime-off arm describes that ablation only. To')
            print('  compare deployment alternatives, use stock vLLM with a plain export.')
        elif kind == "method":
            print("  !! The baseline arm ran with residue ACTIVE. Every delta below")
            print("  !! is method-vs-method, not method-vs-floor.")

    
    
    
    
    print("\n=== arm identity (from provenance, never from the name) ===")
    print(f"  {'arm':<22}{'kill-switch cell':<18}{'calibration':<22}note")
    recipes: dict[str, list[str]] = {}
    for arm_id in order:
        cell, why = _cell(arm_id)
        rec = _prov(arm_id)
        cal = (rec or {}).get("calibration")
        if cal is None:
            fp = "UNRECORDED"
        elif "error" in cal:
            fp = "unreadable"
        else:
            fp = str(cal.get("recipe_fingerprint", "no-fingerprint"))[:12]
        recipes.setdefault(fp, []).append(arm_id)
        print(f"  {arm_id:<22}{str(cell):<18}{fp:<22}{why}")
    live = {k: v for k, v in recipes.items() if k not in ("UNRECORDED", "unreadable")}
    if len(live) > 1:
        print("\n  !! ARMS SPAN MORE THAN ONE CALIBRATION RECIPE -- this is not one")
        print('  !! table. Calibration recipes can affect the baseline and must be')
        print('  !! matched across arms before interpreting the differences')
        print("  !! reported below.")
        for fp, names in live.items():
            print(f"       {fp}: {', '.join(names)}")
    elif not live:
        print("\n  note: no arm records its calibration recipe, so 'same recipe'")
        print("  is an assumption here, not a check. Export with")
        print("  calibration_provenance.json beside the checkpoint to close this.")
    gate = {}
    if ref not in arms or base not in arms:
        
        
        
        
        
        
        
        
        absent = [x for x in (ref, base) if x not in arms]
        print(f"\n  !! NO VENUE CONDITIONING: {', '.join(absent)} produced no "
              f"data under this --out.")
        print(f"     The gate that decides whether this venue can resolve "
              f"quantisation damage")
        print(f"     DID NOT RUN. Everything above is a scores table and "
              f"nothing more -- it is")
        print(f"     not evidence that any difference in it is real. Run the "
              f"{'/'.join(absent)} arm(s)")
        print(f"     and re-analyze; the analysis costs seconds and no GPU.")
    else:
        # The gate answers "can this VENUE resolve quantisation damage at all",
        # which only means anything when ref/base really are the reference and
        # the unquantised-method floor. Point it at two method arms and it
        # silently tests the method against itself, then refuses to report --
        # four FAILs that look like a dead venue rather than a misuse. Say so.
        canon_ref = ref.startswith("bf16") or "bf16" in ref
        canon_base = base.startswith("plain") or "plain" in base
        if not (canon_ref and canon_base):
            print(f"\n  !! GATE MISUSE: --reference {ref} / --baseline {base} are")
            print(f"     not a (bf16, plain) pair, so what follows is NOT venue")
            print(f"     conditioning -- it is one method arm against another, and")
            print(f"     the 3x bar was never meant for that. Condition the venue")
            print(f"     with bf16 vs plain first; read the section below as a")
            print(f"     plain effect-size table, not as a verdict.")
        print(f"\n=== venue conditioning: is there damage to recover? "
              f"({ref} - {base}) ===")
        print(f"{'subset':<22}{ref:>10}{base:>10}{'loss':>11}{'±SE':>8}"
              f"{'loss/SE':>9}{'verdict':>9}")
        for t in tasks:
            A, B = arms[ref].get(t), arms[base].get(t)
            if not A or not B:
                continue
            n = min(len(A), len(B))
            diffs = [A[i]["score"] - B[i]["score"] for i in range(n)]
            m, se = sum(diffs) / n, bootstrap_se(diffs)
            ratio = m / se if se > 0 else 0.0
            gate[t] = ratio
            print(f"{t:<22}{sum(i['score'] for i in A[:n])/n:>10.4f}"
                  f"{sum(i['score'] for i in B[:n])/n:>10.4f}{m*100:>+10.2f}pp"
                  f"{se*100:>8.2f}{ratio:>9.1f}"
                  f"{('PASS' if ratio > 3 else 'FAIL'):>9}")
        print("  Rule: a method comparison is only reported on PASS subsets.")
        print("  Rank by loss/SE, not loss -- the method recovers a fraction of")
        print("  the loss, so a wide-spread subset is worth less than a tight one.")

    for pair in (a.compare or "").split(";"):
        pair = pair.strip()
        if not pair or "vs" not in pair:
            continue
        x, y = [s.strip() for s in pair.split("vs")]
        if x not in arms or y not in arms:
            print(f"\n[skip] {pair}: missing arm")
            continue
        print(f"\n=== {x} vs {y} (paired, per item) ===")
        print(f"{'subset':<22}{y:>12}{x:>12}{'Δ':>11}{'±SE':>8}{'Δ/SE':>8}{'':>8}")
        # Compute every subset, gated or not. Suppressing the numbers entirely
        # hid a real combined effect (z=+3.72 over four subsets that each sat
        # at 1.2-2.3 sigma) behind four FAIL rows, which reads as "nothing here"
        # -- the opposite of what the data said. Mark the gated ones instead;
        # a marked number the reader must discount beats an absent one they
        # cannot see at all.
        zs, zs_all, never = [], [], []
        for t in tasks:
            A, B = arms[x].get(t), arms[y].get(t)
            if not A or not B:
                continue
            n = min(len(A), len(B))
            diffs = [A[i]["score"] - B[i]["score"] for i in range(n)]
            m, se = sum(diffs) / n, bootstrap_se(diffs)
            z = m / se if se > 0 else 0.0
            # Three states, not two. A subset with no gate entry was never
            # conditioned at all -- typically because the reference or the
            # floor arm has no data for it -- and the old default of 99 scored
            # that as "comfortably passed", folding an unmeasured subset into
            # the headline combined z. Absent is not passed.
            if not gate:
                gated = unconditioned = False
            elif t not in gate:
                gated = unconditioned = True
                never.append(t)
            else:
                gated, unconditioned = gate[t] <= 3, False
            zs_all.append(z)
            if not gated:
                zs.append(z)
            print(f"{t:<22}{sum(i['score'] for i in B[:n])/n:>12.4f}"
                  f"{sum(i['score'] for i in A[:n])/n:>12.4f}{m*100:>+10.2f}pp"
                  f"{se*100:>8.2f}{z:>8.1f}"
                  f"{'  [never conditioned]' if unconditioned else ('  [gated]' if gated else '')}")
        import math
        if never:
            print(f"  {len(never)} subset(s) were NEVER conditioned and are "
                  f"excluded from both totals: {', '.join(never)}.")
            print(f"  Their venue was not shown to resolve quantisation damage,"
                  f" so a difference")
            print(f"  on them is not evidence either way. Run the reference and"
                  f" floor arms on them.")
        if zs:
            print(f"{'combined z (conditioned subsets)':<34}"
                  f"{sum(zs)/math.sqrt(len(zs)):>+32.2f}")
        if zs_all and len(zs_all) != len(zs):
            print(f"{'combined z (ALL subsets, incl. gated)':<34}"
                  f"{sum(zs_all)/math.sqrt(len(zs_all)):>+32.2f}")
            print("  The second line includes subsets whose venue conditioning")
            print("  failed. Quote it only with that caveat -- never on its own.")
    return 0


# --------------------------------------------------------------------- ppl
CORPORA = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1", "test", "text", "\n\n"),
    "c4":        ("allenai/c4", "en", "validation", "text", "\n\n"),
}


def load_corpus(name, explicit_file, data_dir):
    """Prefer a local jsonl; the pod has no outbound network."""
    for cand in ([explicit_file] if explicit_file else []) + \
                [str(pathlib.Path(data_dir) / f"corpus_{name}.jsonl")]:
        f = pathlib.Path(cand)
        if f.exists() and f.stat().st_size:
            rows = [json.loads(l) for l in f.open() if l.strip()]
            return "\n\n".join(r["text"] for r in rows), f"local:{f}"
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")
    path, cfg, split, col, joiner = CORPORA[name]
    from datasets import load_dataset
    ds = load_dataset(path, cfg, split=split)
    if name == "c4":
        ds = ds.select(range(min(2000, len(ds))))
    return joiner.join(x[col] for x in ds), f"hub:{path}/{cfg}"


def cmd_ppl(a) -> int:
    'Teacher-forced perplexity THROUGH the framework under test.'
    install_env(a)
    import math
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config.kernel import KernelConfig

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{a.arm}__ppl_{a.corpus}.jsonl"
    if dest.exists() and not a.force:
        print(f"[kit] {dest.name} already on disk; --force to redo")
        return 0

    text, origin = load_corpus(a.corpus, a.corpus_file, a.data_dir)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    # Non-overlapping windows, deterministic from (corpus, tokenizer, seq_len)
    # alone -- so every arm scores the IDENTICAL windows and the comparison is
    # exactly paired.
    wins = [ids[i:i + a.seq_len]
            for i in range(0, len(ids) - a.seq_len + 1, a.seq_len)][: a.n_windows]
    if not wins:
        raise SystemExit(f"corpus too short: {len(ids)} tokens < seq_len {a.seq_len}")
    print(f"[kit] corpus={a.corpus} ({origin}) tokens={len(ids):,} "
          f"windows={len(wins)} x {a.seq_len}", flush=True)

    extra = {}
    if getattr(a, "cudagraph_mode", "") or getattr(a, "compile_mode", ""):
        from vllm.config import CompilationConfig
        kw = {}
        if getattr(a, "cudagraph_mode", ""):
            kw["cudagraph_mode"] = a.cudagraph_mode
        if getattr(a, "compile_mode", ""):
            kw["mode"] = a.compile_mode
        if getattr(a, "inductor_config", ""):
            kw["inductor_compile_config"] = json.loads(a.inductor_config)
        extra["compilation_config"] = CompilationConfig(**kw)
        print(f"[kit] compilation_config {kw}", flush=True)
    ekw = pinned_engine_kwargs(a, max(a.seq_len + 16, 2048))
    write_provenance(a, a.out, {**ekw, **{k: repr(v) for k, v in extra.items()}},
                     {"venue": f"ppl_{a.corpus}", "n_windows": a.n_windows,
                      "seq_len": a.seq_len})
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              quantization=None if a.bf16 else "modelopt_fp4", **extra, **ekw)

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    try:                                    # newer vLLM wants TokensPrompt
        from vllm import TokensPrompt
        outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins], sp)
    except ImportError:
        outs = llm.generate(prompt_token_ids=wins, sampling_params=sp)

    rows, tot_nll, tot_n = [], 0.0, 0
    for w, o in zip(wins, outs):
        lps = o.prompt_logprobs
        # index 0 has no prefix and therefore no logprob -- it is None, and
        # including it would silently shorten every window by one token.
        nll = 0.0
        n = 0
        for i in range(1, len(w)):
            e = lps[i]
            if not e:
                continue
            v = e.get(w[i])
            if v is None:
                continue
            nll -= (v.logprob if hasattr(v, "logprob") else float(v))
            n += 1
        rows.append({"nll": nll, "n": n})
        tot_nll += nll
        tot_n += n
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ppl = math.exp(tot_nll / max(tot_n, 1))
    print(f"[kit] {a.arm:<16}ppl_{a.corpus:<12}{ppl:>10.4f}   "
          f"nll/tok={tot_nll/max(tot_n,1):.6f}  windows={len(rows)}  tokens={tot_n:,}")
    print(f"[kit] wrote {dest}")
    return 0


def cmd_analyze_ppl(a) -> int:
    """Reads ONLY the jsonl. No GPU, seconds, re-runnable forever."""
    import math
    d = pathlib.Path(a.out)
    arms = {}
    for f in sorted(d.glob(f"*__ppl_{a.corpus}.jsonl")):
        arm = f.name.split("__")[0]
        arms[arm] = [json.loads(l) for l in f.open() if l.strip()]
    if not arms:
        print(f"no *__ppl_{a.corpus}.jsonl under {d}")
        return 1
    order = [x for x in a.order.split(",") if x in arms] + \
            [x for x in sorted(arms) if x not in a.order.split(",")]

    def ppl_of(rows):
        return math.exp(sum(r["nll"] for r in rows) / max(sum(r["n"] for r in rows), 1))

    print(f"\ncorpus={a.corpus}   windows={len(next(iter(arms.values())))}")
    print(f"{'arm':<18}{'ppl':>12}{'nll/tok':>12}{'d vs ref':>12}")
    ref = arms.get(a.reference)
    for arm in order:
        rows = arms[arm]
        p = ppl_of(rows)
        nt = sum(r["nll"] for r in rows) / max(sum(r["n"] for r in rows), 1)
        dv = f"{p - ppl_of(ref):+.4f}" if ref else "-"
        print(f"{arm:<18}{p:>12.4f}{nt:>12.6f}{dv:>12}")

    # Paired over windows: same corpus, same tokenizer, same seq_len means
    # window i is literally the same token span in every arm.
    for spec in [s.strip() for s in a.compare.split(";") if s.strip()]:
        x, y = [t.strip() for t in spec.split("vs")]
        if x not in arms or y not in arms:
            continue
        ax, ay = arms[x], arms[y]
        if len(ax) != len(ay):
            print(f"\n{spec}: window counts differ ({len(ax)} vs {len(ay)}) -- skipped")
            continue
        diffs = [(rx["nll"] / max(rx["n"], 1)) - (ry["nll"] / max(ry["n"], 1))
                 for rx, ry in zip(ax, ay)]
        m = sum(diffs) / len(diffs)
        se = bootstrap_se(diffs)
        print(f"\n{spec}: d(nll/tok) = {m:+.6f} +- {se:.6f}   "
              f"z = {m/se if se else 0:+.2f}   (negative = {x} better)")
    return 0


# ------------------------------------------------------------------- task
# Length-normalised accuracy is the published convention for these two, so a
# number scored the other way is not comparable with the literature.
NORM_TASKS = {"arc_challenge", "hellaswag"}


def _mc_scores(llm, tok, items, sp_cls):
    """Summed continuation logprob for every (item, choice).

    Only the continuation is scored; the shared context is excluded. Scoring
    the whole string would let a long context dominate and make the choices
    nearly indistinguishable.
    """
    flat, spans = [], []
    for it in items:
        idx = []
        for ch in it["choices"]:
            if "\u0000" in ch:              # per-choice context (winogrande)
                head, cont = ch.split("\u0000", 1)
                ctx = it["context"] + head
            else:
                ctx, cont = it["context"], ch
            ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
            full_ids = tok(ctx + cont, add_special_tokens=False)["input_ids"]
            idx.append(len(flat))
            flat.append((full_ids, len(ctx_ids)))
        spans.append(idx)
    sp = sp_cls(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    try:
        from vllm import TokensPrompt
        outs = llm.generate([TokensPrompt(prompt_token_ids=f[0]) for f in flat], sp)
    except ImportError:
        outs = llm.generate(prompt_token_ids=[f[0] for f in flat], sampling_params=sp)
    tot = []
    for (ids, ncat), o in zip(flat, outs):
        lps, s, n = o.prompt_logprobs, 0.0, 0
        for i in range(max(ncat, 1), len(ids)):
            e = lps[i]
            if not e:
                continue
            v = e.get(ids[i])
            if v is None:
                continue
            s += v.logprob if hasattr(v, "logprob") else float(v)
            n += 1
        
        
        
        
        
        tot.append((s, max(n, 1), len(ids)))
    return spans, tot


def cmd_task(a) -> int:
    """Downstream benchmarks: GSM8K plus the four-task commonsense row."""
    install_env(a)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config.kernel import KernelConfig
    from datasets import load_dataset

    out_dir = pathlib.Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [x.strip() for x in a.tasks.split(",") if x.strip()]
    todo = [x for x in tasks
            if not (out_dir / f"{a.arm}__{x}.jsonl").exists() or a.force]
    if not todo:
        print(f"[kit] {a.arm}: all {len(tasks)} tasks already on disk")
        return 0
    for x in todo:
        if x not in DOWNSTREAM:
            raise SystemExit(f"unknown task {x!r}; known: {sorted(DOWNSTREAM)}")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    a._venue = "task"
    ekw = pinned_engine_kwargs(a, a.max_model_len)
    write_provenance(a, a.out, ekw, {"venue": "downstream", "tasks": todo,
                                     "max_gen": {t: a.max_gen for t in todo}})
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              quantization=None if a.bf16 else "modelopt_fp4", **ekw)

    for name in todo:
        path, cfg, split, kind = DOWNSTREAM[name]
        ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
        if a.limit:
            ds = ds.select(range(min(a.limit, len(ds))))
        items = [normalise_downstream(name, dict(r)) for r in ds]

        rows = []
        if kind == "gen":
            
            
            
            
            
            
            
            
            
            
            sp = SamplingParams(temperature=0.0, max_tokens=a.max_gen,
                                seed=a.seed, stop=STOPS.get(name))
            outs = llm.generate([it["context"] for it in items], sp)
            for it, o in zip(items, outs):
                pred = o.outputs[0].text
                rows.append({"score": SCORERS[name](pred, it["answers"]),
                             "pred": pred[:200], "answers": it["answers"][:1],
                             "n_prompt": len(o.prompt_token_ids),
                             "n_gen": len(o.outputs[0].token_ids)})
        else:
            spans, tot = _mc_scores(llm, tok, items, SamplingParams)
            for it, idx in zip(items, spans):
                raw = [tot[i][0] for i in idx]
                norm = [tot[i][0] / tot[i][1] for i in idx]
                pick = max(range(len(raw)), key=lambda k: raw[k])
                pick_n = max(range(len(norm)), key=lambda k: norm[k])
                use = pick_n if name in NORM_TASKS else pick
                rows.append({"score": 1.0 if use == it["gold"] else 0.0,
                             "n_prompt": max(tot[i][2] for i in idx),
                             "n_gen": 0,
                             "acc": 1.0 if pick == it["gold"] else 0.0,
                             "acc_norm": 1.0 if pick_n == it["gold"] else 0.0,
                             "gold": it["gold"], "pick": use})
        (out_dir / f"{a.arm}__{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        s = sum(r["score"] for r in rows) / len(rows)
        extra = ""
        if kind == "mc":
            extra = (f"  acc={sum(r['acc'] for r in rows)/len(rows):.4f}"
                     f"  acc_norm={sum(r['acc_norm'] for r in rows)/len(rows):.4f}")
        print(f"[kit] {a.arm:<14}{name:<16}{'acc_norm' if name in NORM_TASKS else 'acc':<9}"
              f"{s:.4f}  n={len(rows)}{extra}", flush=True)
    print(f"[kit] wrote {out_dir}/{a.arm}__*.jsonl")
    return 0


# -------------------------------------------------------------- provenance












def pinned_engine_kwargs(a, max_model_len: int) -> dict:
    from vllm.config.kernel import KernelConfig
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    _short = getattr(a, "cmd", None) == "task" or getattr(a, "_venue", "") == "task"
    mnbt = (getattr(a, "max_num_batched_tokens", None)
            or (2048 if _short else max_model_len))
    return dict(
        enforce_eager=bool(getattr(a, "enforce_eager", False)),   # 1
        max_num_batched_tokens=int(mnbt),                         # 2
        # Knob 3. Default OFF because autotune is pure run-to-run noise for a
        # comparison table -- but OFF is also not what production serves, which
        # is the same failure shape as measuring a floor nobody deploys. The
        # flag exists to MEASURE whether that distinction costs anything: if
        # tuned and untuned score identically, knob 3 can be relaxed to
        # 'prewarm once, then fixed' and evaluation can use the production
        # configuration; if they differ, the prewarm is mandatory.
        kernel_config=KernelConfig(
            enable_flashinfer_autotune=bool(getattr(a, 'flashinfer_autotune', False))),  # 3
        seed=a.seed,                                              # 4
        # 5: NVFP4 GEMM backend deliberately NOT forced -- forcing it distorts
        #    baselines, and vLLM>=0.26 ignores the old env anyway.
        max_model_len=max_model_len,                              # 6
        gpu_memory_utilization=a.gpu_mem_util,                    # 7
        # 8: batch_size n/a -- every prompt is submitted in one generate() call
        max_num_seqs=a.max_num_seqs,                              # 10
        # 11: VLLM_DISABLE_COMPILE_CACHE is set in install_env(), before torch
        #     is imported; it cannot be passed as an engine kwarg.
    )


def write_provenance(a, out_dir, engine_kwargs: dict, extra: dict | None = None):
    """One json per arm, recording what the run's numbers are conditional on.

    Knob 9 is "one machine, one venv" -- not enforceable from inside a single
    run, only recordable, which is precisely why it must be recorded. A table
    is comparable when every arm's provenance agrees; that check costs a diff
    and is the only thing standing between us and comparing across a vLLM
    upgrade without noticing.
    """
    import platform
    rec = {"arm": a.arm, "model": a.model, "cmd": sys.argv[1:],
           "knobs": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                         else repr(v)) for k, v in engine_kwargs.items()},
           "env": {k: os.environ.get(k) for k in sorted(
               [e for e in os.environ if e.startswith(("RESIDUE_", "VLLM_"))])},
           "python": platform.python_version(), "platform": platform.platform()}
    
    
    
    
    
    
    
    
    rec["calibration"] = None
    try:
        cp = pathlib.Path(a.model) / "calibration_provenance.json"
        if cp.exists():
            rec["calibration"] = json.loads(cp.read_text())
    except Exception as exc:  # noqa: BLE001
        rec["calibration"] = {"error": f"{type(exc).__name__}: {exc}"}
    for mod in ("vllm", "torch", "flashinfer", "transformers"):
        try:
            rec.setdefault("versions", {})[mod] = __import__(mod).__version__
        except Exception:
            rec.setdefault("versions", {})[mod] = None
    try:
        import subprocess
        rec["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=a.repo or ".", capture_output=True,
            text=True).stdout.strip() or None
    except Exception:
        rec["git_head"] = None
    try:
        import torch
        rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        rec["gpu"] = None
    if extra:
        rec.update(extra)
    p = pathlib.Path(out_dir) / f"{a.arm}__provenance.json"
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"[kit] provenance -> {p}", flush=True)


# ---------------------------------------------------------------- selftest
def cmd_selftest(a) -> int:
    'A-B-A determinism gate. Run BEFORE any table whose numbers get published.'
    import subprocess
    import tempfile
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model_b = a.model_b or a.model
    if not a.model_b:
        print("[kit] WARNING: no --model-b, so B is the same checkpoint as A. "
              "The gate still tests queue position, but a different checkpoint "
              "is the stronger perturbation. Pass one.", flush=True)
    # B must be the same ARCHITECTURE as A, and that is not a detail -- it is
    # the difference between a gate and a formality. The cache is keyed on the
    # shape of the compute graph, so two checkpoints of different shape get
    # different keys, never collide, and the gate passes trivially while
    # proving nothing. Different WEIGHTS, same SHAPE is the pair that shares a
    # cache entry, and sharing one is the whole phenomenon being tested.
    #
    # Checked rather than documented, because a gate that can pass for the
    # wrong reason is worse than no gate: it produces a PASS line that reads
    # exactly like a real one.
    def _shape(p):
        try:
            c = json.load(open(pathlib.Path(p) / "config.json"))
        except Exception:
            return None
        return tuple(c.get(k) for k in ("hidden_size", "num_hidden_layers",
                                        "num_attention_heads", "intermediate_size"))
    sa, sb = _shape(a.model), _shape(model_b)
    if sa and sb and sa != sb:
        print("[kit] WARNING: --model and --model-b have DIFFERENT shapes")
        print(f"[kit]   A {sa}")
        print(f"[kit]   B {sb}")
        print("[kit] They will not share a torch.compile cache entry, so this "
              "gate cannot see queue contamination and a PASS below means "
              "nothing. Use two checkpoints of the same architecture with "
              "different weights.", flush=True)
    pool = pathlib.Path(tempfile.mkdtemp(prefix="evalkit_selftest_pool_"))

    def one(arm, model, phase):
        cmd = [sys.executable, os.path.abspath(__file__), "run",
               "--model", model, "--arm", arm, "--data-dir", a.data_dir,
               "--out", str(out), "--tasks", a.task, "--force",
               "--max-model-len", str(a.max_model_len),
               "--max-num-seqs", str(a.max_num_seqs),
               "--gpu-mem-util", str(a.gpu_mem_util), "--seed", str(a.seed)]
        if a.repo:
            cmd += ["--repo", a.repo]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.max_num_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(a.max_num_batched_tokens)]
        env = dict(os.environ)
        # The gate must be able to run WITH the cache on demand -- that is how
        # you demonstrate it catches something, and how you measure what the
        # cache is worth. Default stays off, like every other eval.
        if a.allow_cache:
            env["RESIDUE_EVALKIT_ALLOW_CACHE"] = "1"
            env["VLLM_CACHE_ROOT"] = str(pool / phase)
        print(f"[kit] selftest >>> {arm}  phase={phase}  ({model})", flush=True)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"[kit] selftest: arm {arm} failed rc={r.returncode}")
        return r.returncode

    tag = "cacheON" if a.allow_cache else "nocache"
    a1, b0, a2 = f"_st{tag}_A1", f"_st{tag}_B", f"_st{tag}_A2"
    # phase1: A alone on a cold pool.  phase2: B seeds a second cold pool,
    # then A runs into it. Same A, different queue position.
    for arm, model, phase in ((a1, a.model, "phase1"),
                              (b0, model_b, "phase2"),
                              (a2, a.model, "phase2")):
        if one(arm, model, phase) != 0:
            return 2

    f1 = out / f"{a1}__{a.task}.jsonl"
    f2 = out / f"{a2}__{a.task}.jsonl"
    if not (f1.exists() and f2.exists()):
        print("[kit] selftest: FAIL -- an arm produced no jsonl")
        return 2
    r1 = [json.loads(l) for l in f1.read_text().splitlines() if l.strip()]
    r2 = [json.loads(l) for l in f2.read_text().splitlines() if l.strip()]
    if len(r1) != len(r2):
        print(f"[kit] selftest: FAIL -- item counts differ {len(r1)} vs {len(r2)}")
        return 3

    # Compare the TEXT, not only the score. Two different generations can score
    # the same by luck, and that would report a leaking queue as clean.
    n_score = sum(1 for x, y in zip(r1, r2) if x["score"] != y["score"])
    n_pred = sum(1 for x, y in zip(r1, r2) if x["pred"] != y["pred"])
    m1 = sum(x["score"] for x in r1) / len(r1)
    m2 = sum(x["score"] for x in r2) / len(r2)
    print()
    print(f"[kit] selftest  task={a.task}  n={len(r1)}  cache="
          f"{'ALLOWED' if a.allow_cache else 'disabled'}  interposed={model_b}")
    print(f"[kit]   A1 mean {m1:.6f}   A2 mean {m2:.6f}   delta "
          f"{(m2 - m1) * 100:+.4f}pp")
    print(f"[kit]   items with a different score      : {n_score}/{len(r1)}")
    print(f"[kit]   items with a different generation : {n_pred}/{len(r1)}")
    if n_score == 0 and n_pred == 0:
        print("[kit] SELFTEST PASS -- identical through an interposed run. "
              "A table built under this configuration is trustworthy.")
        return 0
    print("[kit] SELFTEST FAIL -- the queue leaks: an arm's number depends on "
          "what ran before it. Do NOT publish a table from this configuration.")
    return 3


# ------------------------------------------------------------- env-check
# The environment is knob 9, the one knob a single run cannot enforce, only
# record. This makes it checkable BEFORE two hours of GPU time rather than
# after: it prints the contract, and with --expect <provenance.json> it
# diffs against a run you already trust and exits non-zero on any difference.
def cmd_envcheck(a) -> int:
    import platform
    got = {}
    for mod in ("vllm", "torch", "flashinfer", "transformers"):
        try:
            got[mod] = __import__(mod).__version__
        except Exception as e:
            got[mod] = f"MISSING ({type(e).__name__})"
    try:
        import torch
        got["cuda"] = torch.version.cuda
        got["gpu"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "NO CUDA DEVICE")
        got["n_gpu"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as e:
        got["gpu"] = f"MISSING ({type(e).__name__})"
    got["python"] = platform.python_version()
    print("[kit] environment")
    for k, v in got.items():
        print(f"[kit]   {k:14s} {v}")
    bad = [k for k, v in got.items() if isinstance(v, str) and v.startswith(("MISSING", "NO CUDA"))]
    if bad:
        print(f"[kit] ENV FAIL -- unusable: {', '.join(bad)}")
        return 2
    if a.expect:
        ref = json.loads(pathlib.Path(a.expect).read_text())
        refv = dict(ref.get("versions") or {})
        refv["gpu"] = ref.get("gpu")
        diffs = [(k, refv[k], got.get(k)) for k in refv
                 if refv[k] is not None and refv[k] != got.get(k)]
        if diffs:
            print(f"[kit] ENV DIFFERS from {a.expect}:")
            for k, want, have in diffs:
                print(f"[kit]   {k:14s} expected {want}  got {have}")
            print("[kit] A table spanning these two environments is not a "
                  "comparison. Re-run the WHOLE table, never one cell.")
            return 3
        print(f"[kit] env matches {a.expect}")
    return 0


# ------------------------------------------------------- venue dispatch
# A manifest declares `venue:`, and the suite drives the matching subcommand.
# Without this the runner covered LongBench only, so "run the official
# evaluation" meant one of four venues plus a memory of how to invoke the rest
# -- which is the same hand-assembly the suite exists to abolish, just moved up
# one level. Every venue a manifest can name is one fewer thing that has to be
# remembered correctly at 2am.
def _argv_longbench(man, root, arm, out, a):
    argv = ["run", "--model", str(root / arm["path"]), "--arm", arm["id"],
            "--data-dir", a.data_dir, "--out", str(out),
            "--max-model-len", str(man.get("max_model_len", 32768)),
            "--max-num-seqs", str(man.get("max_num_seqs", 8)),
            "--gpu-mem-util", str(a.gpu_mem_util),
            "--tasks", man.get("tasks", DEFAULT_TASKS)]
    if man.get("max_num_batched_tokens"):
        argv += ["--max-num-batched-tokens", str(man["max_num_batched_tokens"])]
    if man.get("limit"):
        argv += ["--limit", str(man["limit"])]
    return argv


def _argv_ppl(man, root, arm, out, a):
    return ["ppl", "--model", str(root / arm["path"]), "--arm", arm["id"],
            "--out", str(out), "--data-dir", a.data_dir,
            "--corpus", man.get("corpus", "wikitext2"),
            "--seq-len", str(man.get("seq_len", 2048)),
            "--n-windows", str(man.get("n_windows", 128)),
            "--max-num-seqs", str(man.get("max_num_seqs", 8)),
            "--gpu-mem-util", str(a.gpu_mem_util)]


def _argv_task(man, root, arm, out, a):
    argv = ["task", "--model", str(root / arm["path"]), "--arm", arm["id"],
            "--out", str(out),
            "--tasks", man.get("tasks", "gsm8k,arc_challenge,hellaswag,piqa,winogrande"),
            "--max-model-len", str(man.get("max_model_len", 4096)),
            "--max-num-seqs", str(man.get("max_num_seqs", 8)),
            "--gpu-mem-util", str(a.gpu_mem_util)]
    if man.get("limit"):
        argv += ["--limit", str(man["limit"])]
    return argv


def _argv_external(name):
    """These venues have a driver; it is just not this one.

    This used to end with "the driver is a separate program and is not built
    yet", which was true when written and stopped being true when
    lcb_driver shipped. A reader hitting it concluded the venue was
    unavailable and stopped -- the same defect the README carried, found the
    same way, by someone running the shipped code without having written it.

    So the refusal now routes rather than dead-ends. It still refuses: the
    scorer lives in a separate environment for a reason (lighteval's
    transformers floor is years ahead of what vLLM pins), and pretending
    otherwise from inside this process would import the wrong stack.
    """
    def f(man, root, arm, out, a):
        raise NotImplementedError(
            f"venue '{name}' is not scored by this kit -- it needs lighteval, "
            f"which cannot co-install with vLLM and lives in "
            f".dev/envs/lighteval/. USE THE DRIVER THAT DOES:\n"
            f"    python -m benchmarks.lcb_driver --manifest <this manifest> "
            f"--models-root <dir> --arms <a,b> --out <dir>\n"
            f"It launches one server per arm through the production "
            f"attachment chain and refuses to score a cache replay. See the "
            f"LiveCodeBench section of benchmarks/README.md.")
    return f


VENUES = {
    "longbench": _argv_longbench,
    "ppl": _argv_ppl,
    "task": _argv_task,
    # Declarable so the arm list, decode settings and budget ladder live in one
    # reviewable file from the start rather than being invented alongside the
    # driver. Naming them here and refusing loudly beats omitting them, which
    # would read as "these venues do not exist".
    "lcb": _argv_external("lcb"),
    "aime": _argv_external("aime"),
}


# ----------------------------------------------------------------- suite
def cmd_suite(a) -> int:
    """Run the whole official table as ONE command, in one sitting.

    This exists because the alternative is what actually happened: the table
    got assembled by hand-written shell scripts, one per session, over twenty
    hours -- and then two of those outputs were compared as if they were one
    table, across a GPU change, a config change and a code change. Every
    reproducibility failure this project has had has that shape. A suite is
    the fix, not because it is tidier, but because it makes "assembled across
    time" structurally impossible: one driver process, one code version, one
    environment, gate first, arms in sequence, table at the end.

    Each arm runs as a subprocess, because vLLM cannot swap checkpoints
    in-process. That is the same isolation a real table has, so the gate --
    which measures exactly the cross-process leakage that isolation admits --
    tests the right thing.
    """
    import subprocess
    import time
    man = json.loads(pathlib.Path(a.manifest).read_text()) \
        if a.manifest.endswith(".json") else _load_yamlish(a.manifest)
    root = pathlib.Path(a.models_root).resolve()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = man["arms"]
    missing = [x["id"] for x in arms if not (root / x["path"]).exists()]
    if missing and not a.skip_missing:
        print(f"[kit] SUITE ABORT -- {len(missing)} arm(s) not under {root}:")
        for m in missing:
            print(f"[kit]   {m}: {next(x['path'] for x in arms if x['id'] == m)}")
        print("[kit] Export them first (see benchmarks/README.md), or pass "
              "--skip-missing to run the subset you have. A partial table is "
              "fine; a table with a silently absent arm is not.")
        return 2
    if missing:
        print(f"[kit] SKIPPING {len(missing)} absent arm(s): {', '.join(missing)}")
        arms = [x for x in arms if x["id"] not in missing]

    # The manifest names a reference and a baseline, and analyze's venue
    # conditioning is meaningless without both. Check it HERE, against the arms
    # that will actually run, because the alternative is finding out after the
    # GPU time: analyze skips conditioning when an arm is absent, and until the
    # branch above existed it did so in silence. --skip-missing can drop the
    # reference arm too, which is how a "partial table" quietly becomes an
    # unconditioned one.
    ids = {x["id"] for x in arms}
    for role in ("reference", "baseline"):
        name = man.get(role, {"reference": "bf16", "baseline": "plain"}[role])
        if name not in ids:
            print(f"[kit] SUITE ABORT -- manifest {role} '{name}' is not among "
                  f"the arms that will run.")
            print(f"[kit] Without it there is no venue conditioning, and the "
                  f"table would be a scores list that reads like a result. "
                  f"Fix the manifest, or export the arm.")
            return 2

    def kit(*argv, env=None):
        return subprocess.run([sys.executable, os.path.abspath(__file__), *argv],
                              env=env or dict(os.environ)).returncode

    venue = man.get("venue", "longbench")
    if venue not in VENUES:
        print(f"[kit] SUITE ABORT -- unknown venue '{venue}'. "
              f"Known: {', '.join(sorted(VENUES))}")
        return 2
    try:
        VENUES[venue](man, root, arms[0], out, a)
    except NotImplementedError as e:
        # Fail here, before env-check and before the gate: this is a property
        # of the manifest, and it costs nothing to discover.
        print(f"[kit] SUITE ABORT -- {e}")
        return 2
    print(f"[kit] suite '{man.get('name', a.manifest)}'  venue={venue}  "
          f"{len(arms)} arm(s)  -> {out}", flush=True)
    if kit("env-check") != 0:
        print("[kit] SUITE ABORT -- environment unusable")
        return 2

    # The gate, first and blocking. A suite that runs the table anyway on a
    # leaking queue produces exactly the artifact this whole file exists to
    # prevent, and it would look completely normal.
    if not a.no_gate:
        g = man.get("gate") or {}
        ma = g.get("model_a") or arms[0]["id"]
        mb = g.get("model_b") or next((x["id"] for x in arms if x["id"] != ma), None)
        pa = next((x["path"] for x in arms if x["id"] == ma), None)
        pb = next((x["path"] for x in arms if x["id"] == mb), None) if mb else None
        # Bare next() raised StopIteration here, which surfaces as an opaque
        # crash rather than a diagnosis -- and it fires exactly when
        # --skip-missing has dropped the gate's own checkpoint, i.e. when the
        # user most needs to be told which arm went missing.
        if pa is None:
            print(f"[kit] SUITE ABORT -- gate model_a '{ma}' is not among the "
                  f"arms that will run.")
            return 2
        if pb is None:
            # Silently falling back to A-A would still print SELFTEST PASS,
            # and that PASS cannot see queue contamination at all -- the one
            # thing the gate exists for. Refuse instead of weakening.
            print(f"[kit] SUITE ABORT -- the gate needs a SECOND checkpoint "
                  f"(gate.model_b); '{mb}' is absent.")
            print(f"[kit] Without it this degrades to an A-A repeat, which "
                  f"cannot see queue contamination and would still print PASS.")
            return 2
        gargv = ["selftest", "--model", str(root / pa), "--data-dir", a.data_dir,
                 "--out", str(out / "_gate"), "--task", g.get("task", "2wikimqa"),
                 "--limit", str(g.get("limit", 100)),
                 "--gpu-mem-util", str(a.gpu_mem_util),
                 # The gate must certify the configuration the TABLE runs. With
                 # the manifest's engine knobs left off, it certified evalkit's
                 # defaults instead -- a different compile range, hence a
                 # different execution path, hence a PASS about something else.
                 "--max-model-len", str(man.get("max_model_len", 32768)),
                 "--max-num-seqs", str(man.get("max_num_seqs", 8))]
        if man.get("max_num_batched_tokens"):
            gargv += ["--max-num-batched-tokens", str(man["max_num_batched_tokens"])]
        if pb:
            gargv += ["--model-b", str(root / pb)]
        if a.repo:
            gargv += ["--repo", a.repo]
        print(f"[kit] === gate: {ma} vs {mb} ===", flush=True)
        if kit(*gargv) != 0:
            print("[kit] SUITE ABORT -- gate failed. Nothing from this "
                  "configuration is publishable; fix the environment, do not "
                  "rerun the table hoping it passes.")
            return 3

    t0 = time.time()
    failed = []
    for i, arm in enumerate(arms, 1):
        argv = VENUES[venue](man, root, arm, out, a)
        argv += list(arm.get("flags") or [])
        if a.repo:
            argv += ["--repo", a.repo]
        print(f"[kit] === arm {i}/{len(arms)}: {arm['id']} "
              f"({int(time.time() - t0)}s elapsed) ===", flush=True)
        if kit(*argv) != 0:
            failed.append(arm["id"])
            print(f"[kit] arm {arm['id']} FAILED", flush=True)

    print(f"[kit] === table ({int(time.time() - t0)}s) ===", flush=True)
    order = ",".join(x["id"] for x in arms)
    kit("analyze", "--out", str(out),
        "--reference", man.get("reference", "bf16"),
        "--baseline", man.get("baseline", "plain"),
        "--order", order, "--compare", man.get("compare", ""))
    if failed:
        # Loud, and non-zero: a table missing arms reads as a complete table
        # to anyone who did not watch it run.
        print(f"[kit] SUITE INCOMPLETE -- {len(failed)} arm(s) failed: "
              f"{', '.join(failed)}. The table above is missing them.")
        return 4
    print("[kit] SUITE COMPLETE -- gate passed, every arm ran, one code "
          "version, one environment.")
    return 0


def _load_yamlish(path):
    """Manifest loader with no PyYAML dependency.

    The kit's whole point is that it installs nowhere and imports nothing of
    ours; taking a yaml dependency for a forty-line manifest would undo that
    on the one machine (the pod) that cannot pip install. Supports exactly the
    subset the shipped manifests use.
    """
    try:
        import yaml
        return yaml.safe_load(pathlib.Path(path).read_text())
    except ImportError:
        pass
    doc, cur = {}, None
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ind = len(line) - len(line.lstrip())
        s = line.strip()
        if s.startswith("- "):                       # list item -> dict entry
            item, s = {}, s[2:]
            doc.setdefault(cur, []).append(item)
            for part in _split_inline(s):
                k, v = part.split(":", 1)
                item[k.strip()] = _scalar(v.strip())
        elif ind == 0 and s.endswith(":"):
            cur = s[:-1]
            doc.setdefault(cur, {} if cur != "arms" else [])
        elif ind == 0:
            k, v = s.split(":", 1)
            doc[k.strip()] = _scalar(v.strip())
        else:
            k, v = s.split(":", 1)
            if isinstance(doc.get(cur), dict):
                doc[cur][k.strip()] = _scalar(v.strip())
            elif isinstance(doc.get(cur), list) and doc[cur]:
                doc[cur][-1][k.strip()] = _scalar(v.strip())
    return doc


def _split_inline(s):
    """Split `a: 1, b: [x, y]` on commas that are not inside brackets."""
    out, depth, buf = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf); buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return out


def _scalar(v):
    if v.startswith("[") and v.endswith("]"):
        return [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v.strip("'\"")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="materialise subsets to local jsonl")
    f.add_argument("--data-dir", required=True)
    f.add_argument("--tasks", default=DEFAULT_TASKS)
    f.set_defaults(fn=cmd_fetch)

    r = sub.add_parser("run", help="evaluate one arm")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repo", default="", help="repo root, for quant/ + csrc")
    r.add_argument("--tasks", default=DEFAULT_TASKS)
    r.add_argument("--bf16", action="store_true")
    r.add_argument("--plain", action="store_true")
    r.add_argument("--prefill-residue-off", action="store_true")
    r.add_argument("--decode-mext-off", action="store_true",
                   help="NOT IMPLEMENTABLE today -- errors out; see install_env()")
    r.add_argument("--residue-off", action="store_true",
                   help="kill residue at EVERY M (the plain floor, from the "
                        "residue checkpoint itself)")
    r.add_argument("--prefill-only", type=int, metavar="FOLD_BAND",
                   help="kill residue in the DECODE band (M <= FOLD_BAND), "
                        "leaving prefill-only -- the 2x2's fourth cell, "
                        "without which prefill+decode cannot be told apart "
                        "from max(prefill, decode). Pass the fold band, "
                        "typically 64.")
    r.add_argument("--max-model-len", type=int, default=32768)
    r.add_argument("--max-num-seqs", type=int, default=8)
    r.add_argument("--flashinfer-autotune", action="store_true",
                   help="DIAGNOSTIC: run with FlashInfer autotune ON (the production setting). Off by default -- knob 3. A table must not mix the two.")
    r.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="knob 2 -- MUST be explicit; any fixed value works, what matters is that it is identical across the table")
    # knob 7. ONE default across every subcommand. They used to be 0.85 /
    # 0.85 / 0.62 / 0.55 / 0.55, which makes "unset" not a value -- the
    # same defect as the max_num_seqs 8-vs-4 split that silently divided
    # one table across two batch compositions. 0.55 is what every table on
    # record used; 0.85 on a 178GiB card asks for 151GiB and is the most
    # aggressive KV allocation available, which is not a default an
    # unattended task should inherit.
    r.add_argument("--gpu-mem-util", type=float, default=0.55)
    r.add_argument("--mext-max-m", type=int, default=64)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--force", action="store_true")
    # Diagnostic only. eager and non-eager are different execution paths, so
    # an eager arm is not comparable with the scored table -- it exists to
    # ask whether an effect survives without CUDA graph capture.
    r.add_argument("--enforce-eager", action="store_true")
    r.set_defaults(fn=cmd_run)

    z = sub.add_parser("analyze", help="table + conditioning gate + paired tests")
    z.add_argument("--out", required=True)
    z.add_argument("--reference", default="bf16")
    z.add_argument("--baseline", default="plain")
    z.add_argument("--order", default="bf16,plain,prefill_off,uniform,b45,b80")
    z.add_argument("--compare", default="b45 vs prefill_off; b45 vs uniform")
    z.set_defaults(fn=cmd_analyze)

    q = sub.add_parser("ppl", help="teacher-forced perplexity (prefill-only)")
    q.add_argument("--model", required=True)
    q.add_argument("--arm", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--data-dir", default=".")
    q.add_argument("--repo", default="")
    q.add_argument("--corpus", default="wikitext2", choices=sorted(CORPORA))
    q.add_argument("--corpus-file", default="")
    q.add_argument("--seq-len", type=int, default=2048)
    q.add_argument("--n-windows", type=int, default=128)
    q.add_argument("--bf16", action="store_true")
    q.add_argument("--plain", action="store_true")
    q.add_argument("--prefill-residue-off", action="store_true")
    q.add_argument("--decode-mext-off", action="store_true")
    q.add_argument("--residue-off", action="store_true")
    
    
    
    q.add_argument("--max-num-seqs", type=int, default=8)
    q.add_argument("--flashinfer-autotune", action="store_true",
                   help="DIAGNOSTIC: run with FlashInfer autotune ON (the production setting). Off by default -- knob 3. A table must not mix the two.")
    q.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="knob 2 -- MUST be explicit; any fixed value works, what matters is that it is identical across the table")
    q.add_argument("--gpu-mem-util", type=float, default=0.55)
    q.add_argument("--mext-max-m", type=int, default=64)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--force", action="store_true")
    # Diagnostic: which capture mode introduces a difference between two arms
    # that are numerically equivalent in eager. NONE/PIECEWISE/FULL/
    # FULL_DECODE_ONLY/FULL_AND_PIECEWISE.
    q.add_argument("--cudagraph-mode", default="")
    # NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE / VLLM_COMPILE.
    # Isolates whether a difference comes from tracing or from inductor codegen.
    q.add_argument("--compile-mode", default="")
    # JSON dict merged into inductor_compile_config. Exists to test whether a
    # compile-time *benchmark* (not a heuristic) is what makes two nominally
    # equivalent arms differ -- e.g. {"benchmark_combo_kernel": false}.
    q.add_argument("--inductor-config", default="")
    q.set_defaults(fn=cmd_ppl)

    k = sub.add_parser("task", help="downstream benchmarks (gsm8k + commonsense)")
    k.add_argument("--model", required=True)
    k.add_argument("--arm", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--tasks", default="gsm8k,arc_challenge,hellaswag,piqa,winogrande")
    k.add_argument("--repo", default="")
    k.add_argument("--bf16", action="store_true")
    k.add_argument("--plain", action="store_true")
    k.add_argument("--prefill-residue-off", action="store_true")
    k.add_argument("--residue-off", action="store_true")
    k.add_argument("--decode-mext-off", action="store_true")
    k.add_argument("--max-model-len", type=int, default=4096)
    k.add_argument("--max-num-seqs", type=int, default=8)
    k.add_argument("--flashinfer-autotune", action="store_true",
                   help="DIAGNOSTIC: run with FlashInfer autotune ON (the production setting). Off by default -- knob 3. A table must not mix the two.")
    k.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="knob 2 -- MUST be explicit; any fixed value works, what matters is that it is identical across the table")
    k.add_argument("--gpu-mem-util", type=float, default=0.55)
    k.add_argument("--mext-max-m", type=int, default=64)
    k.add_argument("--max-gen", type=int, default=256)
    k.add_argument("--limit", type=int, default=None)
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--enforce-eager", action="store_true")
    k.add_argument("--force", action="store_true")
    k.set_defaults(fn=cmd_task)

    y = sub.add_parser("analyze-ppl", help="ppl table + paired window test")
    y.add_argument("--out", required=True)
    y.add_argument("--corpus", default="wikitext2")
    y.add_argument("--reference", default="bf16")
    y.add_argument("--order", default="bf16,plain,prefill_off,uniform,b45,b50,b60,b80")
    y.add_argument("--compare", default="b45 vs plain; b45 vs uniform")
    y.set_defaults(fn=cmd_analyze_ppl)

    s = sub.add_parser("selftest",
                       help="A-B-A determinism gate; run before any published table")
    s.add_argument("--model", required=True, help="arm A -- run first and last")
    s.add_argument("--model-b", default="",
                   help="a DIFFERENT checkpoint, interposed between the two A "
                        "runs. Without it the gate cannot see contamination.")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--repo", default="")
    s.add_argument("--task", default="2wikimqa",
                   help="one subset; the shortest-prompt one keeps the gate cheap")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--max-model-len", type=int, default=32768)
    s.add_argument("--max-num-seqs", type=int, default=8)
    s.add_argument("--max-num-batched-tokens", type=int, default=None)
    s.add_argument("--gpu-mem-util", type=float, default=0.55)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--allow-cache", action="store_true",
                   help="deliberately run the gate WITH the compile cache, to "
                        "show it catches something and to price the hazard")
    s.set_defaults(fn=cmd_selftest)

    e = sub.add_parser("env-check",
                       help="print the environment contract; --expect diffs it")
    e.add_argument("--expect", default="",
                   help="a provenance.json from a run you trust; non-zero on any diff")
    e.set_defaults(fn=cmd_envcheck)

    w = sub.add_parser("suite",
                       help="THE OFFICIAL RUNNER: env-check + gate + every arm + table")
    w.add_argument("--manifest", required=True, help="benchmarks/suites/*.yaml")
    w.add_argument("--models-root", required=True,
                   help="directory the manifest's arm paths are relative to")
    w.add_argument("--data-dir", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--repo", default="")
    w.add_argument("--gpu-mem-util", type=float, default=0.55)
    w.add_argument("--skip-missing", action="store_true",
                   help="run the arms present instead of aborting; the skipped "
                        "ones are named loudly, because a table with a silently "
                        "absent arm reads as complete")
    w.add_argument("--no-gate", action="store_true",
                   help="DIAGNOSTIC ONLY -- a table produced with this flag is "
                        "not publishable")
    w.set_defaults(fn=cmd_suite)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
