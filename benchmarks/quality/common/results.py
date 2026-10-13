"""CPU-only result records, completeness checks and paired analysis."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import statistics as st
import sys


def input_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def claim_arm_output(a):
    """Each arm is a new experiment; never resume or overwrite its files."""
    directory = pathlib.Path(a.out)
    directory.mkdir(parents=True, exist_ok=True)
    if not a.arm or pathlib.Path(a.arm).name != a.arm or "__" in a.arm:
        raise ValueError("arm must be a filename component without '__'")
    if any(p.name.startswith(a.arm + "__") for p in directory.iterdir()):
        raise ValueError(f"Results already exist for arm {a.arm!r}; use a new output directory/arm. "
                         "--force does not overwrite historical results.")
    # Exclusive creation also prevents two processes from claiming the same arm.
    with (directory / f"{a.arm}__run.json").open("x") as stream:
        json.dump({"model": a.model, "argv": sys.argv[1:]}, stream)
    return directory


def target_logprob_sum(ids, output, start=1):
    """Require every target token in the requested span; no missing-as-zero."""
    if list(output.prompt_token_ids) != list(ids):
        raise ValueError("Returned prompt tokens do not match the scoring input")
    lps = output.prompt_logprobs
    if lps is None or len(lps) != len(ids) or not 1 <= start < len(ids):
        raise ValueError("Incomplete logprob coverage or zero scored tokens")
    values = []
    for i in range(start, len(ids)):
        entry = lps[i]
        value = entry.get(ids[i]) if entry else None
        if value is None:
            raise ValueError(f"Missing target logprob at token {i}")
        value = float(value.logprob if hasattr(value, "logprob") else value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite target logprob at token {i}")
        values.append(value)
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("Nonfinite summed logprob")
    return total, len(values)


def require_output_count(inputs, outputs):
    if len(inputs) != len(outputs):
        raise ValueError(f"Incomplete generation: {len(outputs)} outputs for {len(inputs)} inputs")


def generation_record(item, output):
    if len(output.outputs) != 1:
        raise ValueError("Expected exactly one completion per input")
    completion = output.outputs[0]
    return {"item_id": input_hash(item), "pred": completion.text,
            "prompt_token_ids": list(output.prompt_token_ids),
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
            "n_prompt": len(output.prompt_token_ids), "n_gen": len(completion.token_ids)}


def validate_ppl_rows(rows):
    if not rows:
        raise ValueError("PPL has no scored windows")
    for row in rows:
        if (not isinstance(row.get("n"), int) or row["n"] <= 0
                or not math.isfinite(row["nll"])):
            raise ValueError("PPL requires positive token coverage and finite NLL")
    if not math.isfinite(math.fsum(row["nll"] for row in rows)):
        raise ValueError("Nonfinite aggregate PPL NLL")


def require_paired_ppl(left, right):
    validate_ppl_rows(left)
    validate_ppl_rows(right)
    keys = ("corpus_sha256", "tokenizer_sha256", "window_sha256", "window_index", "n")
    if len(left) != len(right):
        raise ValueError("PPL window counts differ")
    for a, b in zip(left, right):
        if (any(not isinstance(a.get(key), str) or not a[key] for key in keys[:3])
                or any(key not in a or key not in b or a[key] != b[key] for key in keys)):
            raise ValueError("PPL pairing requires matching corpus, tokenizer and token windows")


def bootstrap_se(diffs, n_boot=4000, seed=0):
    r = random.Random(seed)
    n = len(diffs)
    return st.pstdev([sum(diffs[r.randrange(n)] for _ in range(n)) / n
                      for _ in range(n_boot)])


def pair_quality_rows(left, right):
    """Align complete cohorts by identity; never infer pairs from row order."""
    def index(rows):
        result = {}
        for row in rows:
            key = row.get("item_id")
            if not isinstance(key, str) or not key or key in result:
                raise ValueError("Quality pairing requires unique, nonempty item_id values; "
                                 "legacy results without identities cannot be paired")
            score = row.get("score")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("Quality pairing requires finite scores")
            result[key] = row
        return result

    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("Quality pairing requires identical nonempty item cohorts")
    aligned = [b[key] for key in a]
    for x, y in zip(a.values(), aligned):
        if "prompt_token_ids" in x or "prompt_token_ids" in y:
            if (not x.get("prompt_token_ids") or
                    x["prompt_token_ids"] != y.get("prompt_token_ids")):
                raise ValueError("Quality pairing requires identical rendered prompt tokens")
    return list(a.values()), aligned


def cmd_analyze(a) -> int:
    """Reads ONLY the jsonl on disk. No GPU, seconds, re-runnable forever."""
    d = pathlib.Path(a.out)
    def _prov(arm_id):
        p = pathlib.Path(a.out) / f"{arm_id}__provenance.json"
        return json.loads(p.read_text()) if p.exists() else None

    arms, tasks = {}, set()
    for f in sorted(d.glob("*__*.jsonl")):
        arm, task = f.stem.split("__", 1)
        # Early lm-eval runs saved traces alongside scores. New runs use raw/.
        if task == "requests":
            continue
        arms.setdefault(arm, {})[task] = [json.loads(l) for l in f.open() if l.strip()]
        tasks.add(task)
    if not arms:
        print(f"no results under {d}")
        return 1
    tasks = sorted(tasks)
    # Include arms omitted from the requested display order.
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

    # Report token counts to distinguish short and long prefill workloads.
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

    # Use medians to limit the influence of unusually long generations.
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

        # Compare generation lengths for correct and incorrect answers.
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
                # Prefer the recorded cap; infer it only for legacy results.
                declared = ((_prov(x) or {}).get("max_gen") or {}).get(t)
                cap = int(declared) if declared else max(i["n_gen"] for i in rows)
                at_cap = sum(1 for v in bad if v >= cap)
                # Signature A: wrong answers pinned at the ceiling while right
                # answers are far below it -- the venue is scoring truncation.
                suspect = at_cap >= max(3, 0.5 * len(bad)) and mo and mb >= 2 * mo
                # Also flag when both correct and incorrect answers reach the cap.
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

    # Distinguish stock checkpoints from runtime ablations using provenance.
    def _cell(arm_id):
        """Read the residue ablation scope from recorded environment settings."""
        rec = _prov(arm_id)
        if rec is None:
            return None, "no provenance.json -- the cell cannot be determined"
        env = rec.get("env") or {}
        if (env.get("RESIDUE_MXFP4") == "1" or
                rec.get("quantization") in ("modelopt_mxfp4", "residue_mxfp4")):
            return "mxfp4-unknown", ("MXFP4 adapter; residue scope requires a layer audit, "
                                      "not NVFP4 kill-switch inference")
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

    # Report method scope and calibration recipe for each arm.
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
        # Report a missing reference or baseline explicitly.
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
        # The conditioning check expects a BF16 reference and a plain baseline.
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
            if A is None or B is None:
                continue
            A, B = pair_quality_rows(A, B)
            n = len(A)
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
        # Report all paired differences, marking subsets that fail conditioning.
        zs, zs_all, never = [], [], []
        for t in tasks:
            A, B = arms[x].get(t), arms[y].get(t)
            if A is None or B is None:
                continue
            A, B = pair_quality_rows(A, B)
            n = len(A)
            diffs = [A[i]["score"] - B[i]["score"] for i in range(n)]
            m, se = sum(diffs) / n, bootstrap_se(diffs)
            z = m / se if se > 0 else 0.0
            # Keep missing conditioning evidence distinct from a passed check.
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
        validate_ppl_rows(rows)
        return math.exp(sum(r["nll"] for r in rows) / sum(r["n"] for r in rows))

    print(f"\ncorpus={a.corpus}   windows={len(next(iter(arms.values())))}")
    print(f"{'arm':<18}{'ppl':>12}{'nll/tok':>12}{'d vs ref':>12}")
    ref = arms.get(a.reference)
    for rows in arms.values():
        validate_ppl_rows(rows)
        if ref is not None:
            require_paired_ppl(ref, rows)
    for arm in order:
        rows = arms[arm]
        p = ppl_of(rows)
        nt = sum(r["nll"] for r in rows) / sum(r["n"] for r in rows)
        dv = f"{p - ppl_of(ref):+.4f}" if ref else "-"
        print(f"{arm:<18}{p:>12.4f}{nt:>12.6f}{dv:>12}")

    # Paired over windows: same corpus, same tokenizer, same seq_len means
    # window i is literally the same token span in every arm.
    for spec in [s.strip() for s in a.compare.split(";") if s.strip()]:
        x, y = [t.strip() for t in spec.split("vs")]
        if x not in arms or y not in arms:
            continue
        ax, ay = arms[x], arms[y]
        require_paired_ppl(ax, ay)
        diffs = [rx["nll"] / rx["n"] - ry["nll"] / ry["n"]
                 for rx, ry in zip(ax, ay)]
        m = sum(diffs) / len(diffs)
        se = bootstrap_se(diffs)
        print(f"\n{spec}: d(nll/tok) = {m:+.6f} +- {se:.6f}   "
              f"z = {m/se if se else 0:+.2f}   (negative = {x} better)")
    return 0
