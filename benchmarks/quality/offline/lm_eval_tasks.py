"""Thin adapter to the pinned upstream task definitions and vLLM backend."""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

VERSION = '0.4.13'
LONGBENCH_TASKS = 'longbench_passage_retrieval_en,longbench_2wikimqa,longbench_musique,longbench_hotpotqa'
# The primary metric is explicit; upstream also saves every other metric/filter.
METRICS = {'arc_challenge': ('acc_norm', 'none'), 'hellaswag': ('acc_norm', 'none'),
           'piqa': ('acc', 'none'), 'winogrande': ('acc', 'none'),
           'gsm8k': ('exact_match', 'strict-match')}
METRICS.update({task: ('score', 'none') for task in LONGBENCH_TASKS.split(',')})


def scored_rows(result, task):
    """Check upstream completion before translating rows for our paired analysis."""
    from benchmarks.quality.common.results import input_hash
    metric, selected_filter = METRICS[task]
    expected = result['n-samples'][task]['effective']
    samples = [s for s in result['samples'][task] if s['filter'] == selected_filter]
    if not isinstance(expected, int) or expected <= 0 or len(samples) != expected:
        raise ValueError(f'{task}: incomplete upstream samples')
    seen, rows = set(), []
    aggregate = result['results'][task][f'{metric},{selected_filter}']
    for sample in samples:
        identity = sample['doc_id']
        value = sample[metric]
        if identity in seen or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{task}: duplicate sample or invalid metric')
        seen.add(identity)
        item_id = input_hash({'task': task, 'doc': sample['doc'],
                              'arguments': sample['arguments'],
                              'protocol': f'lm-eval-{VERSION}:{metric}:{selected_filter}'})
        rows.append({'item_id': item_id, 'score': value, 'metric': metric,
                     'filter': selected_filter, 'protocol': f'lm-eval-{VERSION}',
                     'sample': sample})
    if not math.isfinite(aggregate) or not math.isclose(
            aggregate, sum(r['score'] for r in rows) / expected, abs_tol=1e-8):
        raise ValueError(f'{task}: aggregate disagrees with saved samples')
    return rows


def validate_wikitext(result):
    """Validate coverage and use upstream's weighted aggregations, never mean PPL."""
    from lm_eval.api.metrics import bits_per_byte, weighted_perplexity
    samples = result['samples']['wikitext']
    expected = result['n-samples']['wikitext']['effective']
    ids = [s['doc_id'] for s in samples]
    if expected <= 0 or len(samples) != expected or len(set(ids)) != expected:
        raise ValueError('wikitext: incomplete or duplicate samples')
    for metric, aggregate in (('word_perplexity', weighted_perplexity),
                              ('byte_perplexity', weighted_perplexity),
                              ('bits_per_byte', bits_per_byte)):
        values = [s[metric] for s in samples]
        if any(s['filter'] != 'none' for s in samples) or any(
                len(v) != 2 or not math.isfinite(v[0]) or v[0] > 0
                or not isinstance(v[1], int) or v[1] <= 0 for v in values):
            raise ValueError(f'wikitext: invalid loglikelihood/coverage for {metric}')
        reported = result['results']['wikitext'][metric + ',none']
        if not math.isfinite(reported) or not math.isclose(reported, aggregate(values), rel_tol=1e-9):
            raise ValueError(f'wikitext: aggregate disagrees with samples for {metric}')


def summarize(a):
    paths = sorted(Path(a.out).glob('*__lm_eval.json'))
    if not paths:
        raise ValueError('No native lm-eval results found')
    for path in paths:
        print(path.name)
        for task, metrics in json.loads(path.read_text())['results'].items():
            print(f"  {task}: " + ', '.join(f'{k}={v}' for k, v in metrics.items()
                                           if ',' in k and '_stderr' not in k))
    return 0


def run(a):
    from benchmarks.quality.common.results import (
        claim_arm_output, target_logprob_sum, require_output_count,
    )
    from benchmarks.quality.common.runtime import (
        install_env, pinned_engine_kwargs, quality_quantization,
        write_provenance, write_effective_engine_config,
    )
    if importlib.metadata.version('lm-eval') != VERSION:
        raise RuntimeError(f'Task evaluation requires lm-eval=={VERSION}; do not patch old model imports')
    tasks = [t.strip() for t in a.tasks.split(',') if t.strip()]
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - (METRICS.keys() | {'wikitext'}):
        raise ValueError(f"Select distinct supported tasks: {sorted(METRICS.keys() | {'wikitext'})}")
    if a.limit is not None and a.limit <= 0:
        raise ValueError('--limit must be positive')
    chat = getattr(a, 'apply_chat_template', False)
    truncation = getattr(a, 'truncation_side', 'left')
    if 'wikitext' in tasks and (chat or a.num_fewshot not in (None, 0)):
        raise ValueError('wikitext rolling likelihood requires no chat template and zero few-shot examples')
    if a.max_gen is not None and a.max_gen <= 0:
        raise ValueError('--max-gen must be positive')
    install_env(a)
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable

    out = claim_arm_output(a)
    raw = out / 'raw'
    raw.mkdir(exist_ok=True)
    engine = pinned_engine_kwargs(a, a.max_model_len)
    write_provenance(a, a.out, engine, {'venue': 'downstream', 'tasks': tasks,
        'scorer': f'lm-eval-{VERSION}', 'num_fewshot': a.num_fewshot,
        'max_gen_override': a.max_gen, 'apply_chat_template': chat,
        'truncation_side': truncation, 'enable_thinking': False})

    generation_audit = {"completions": 0, "length_stops": 0, "empty_outputs": 0}

    class CheckedVLLM(VLLM):
        @staticmethod
        def _parse_logprobs(tokens, outputs, ctxlen):
            target_logprob_sum(tokens, outputs, ctxlen)
            return VLLM._parse_logprobs(tokens, outputs, ctxlen)

        def _model_generate(self, requests, generate=False, sampling_params=None):
            outputs = super()._model_generate(requests, generate, sampling_params)
            require_output_count(requests, outputs)
            # Upstream samples link by prompt tokens/text; keep full raw evidence
            # once, including auxiliary generation from loglikelihood requests.
            with (raw / f'{a.arm}__requests.jsonl').open('a') as stream:
                for ids, output in zip(requests, outputs):
                    if list(output.prompt_token_ids) != list(ids):
                        raise ValueError('Returned prompt does not match requested tokens')
                    row = {'kind': 'generate' if generate else 'loglikelihood',
                           'prompt_token_ids': list(ids), 'outputs': [
                        {'text': o.text, 'token_ids': list(o.token_ids),
                         'finish_reason': o.finish_reason, 'stop_reason': o.stop_reason}
                        for o in output.outputs]}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    if generate:
                        for completion in row['outputs']:
                            generation_audit['completions'] += 1
                            generation_audit['length_stops'] += completion['finish_reason'] == 'length'
                            generation_audit['empty_outputs'] += not completion['text'].strip()
            return outputs

    model = CheckedVLLM(pretrained=a.model, dtype='bfloat16',
        quantization=quality_quantization(a), trust_remote_code=True,
        batch_size=engine.get('max_num_seqs', a.max_num_seqs),
        enable_thinking=False, truncation_side=truncation, **engine)
    write_effective_engine_config(model.model, a)
    result = simple_evaluate(model=model, tasks=tasks,
        task_manager=TaskManager(include_path=a.task_path),
        num_fewshot=a.num_fewshot, limit=a.limit, log_samples=True,
        apply_chat_template=chat,
        gen_kwargs={'max_gen_toks': a.max_gen} if a.max_gen is not None else None,
        random_seed=a.seed, numpy_random_seed=a.seed, torch_random_seed=a.seed,
        fewshot_random_seed=a.seed, bootstrap_iters=0)
    if 'wikitext' in tasks:
        validate_wikitext(result)
    rows = {task: scored_rows(result, task) for task in tasks if task != 'wikitext'}
    with (out / f'{a.arm}__lm_eval.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, default=handle_non_serializable)
    for task, samples in rows.items():
        with (out / f'{a.arm}__{task}.jsonl').open('x') as stream:
            for row in samples:
                stream.write(json.dumps(row, ensure_ascii=False, default=handle_non_serializable) + '\n')
    (out / f'{a.arm}__generation_audit.json').write_text(
        json.dumps(generation_audit, indent=2))
    if generation_audit['length_stops'] or generation_audit['empty_outputs']:
        print(f'[kit] generation audit requires review: {generation_audit}; scores are unchanged')
    print(f'[kit] completed lm-eval {VERSION}: {tasks}; official results in {out}')
    return 0
