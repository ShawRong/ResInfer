"""Validate pinned lighteval artifacts using its actual scheduled documents."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
from pathlib import Path


def identity(doc):
    fields = {k: doc.get(k) for k in ('task_name', 'query', 'choices', 'gold_index', 'instruction')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(report, expected, details, aggregations):
    results = report.get('results')
    if not results or not expected or not set(expected) <= results.keys():
        raise ValueError('Missing expected task results')
    for task, spec in expected.items():
        metrics = results[task]
        if not spec['metrics'] or not set(spec['metrics']) <= metrics.keys():
            raise ValueError(f'{task}: missing required metrics')
        if any(not isinstance(metrics[k], Real) or not math.isfinite(metrics[k])
               for k in spec['metrics']):
            raise ValueError(f'{task}: nonfinite or invalid aggregate')
        wanted = Counter((x['id'], x['num_samples']) for x in spec['docs'])
        actual = Counter()
        values = {key: [] for key in spec['metrics']}
        for row in details.get(task, []):
            doc, response = row['doc'], row['model_response']
            n = doc['num_samples']
            texts = response['text']
            if not isinstance(texts, list) or not isinstance(n, int) or n <= 0 or len(texts) != n:
                raise ValueError(f'{task}: incomplete generations')
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError(f'{task}: empty or invalid generation; check the API/scorer logs')
            scores = row.get('metric') or {}
            for key in values:
                value = scores.get(key)
                if not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f'{task}: missing or invalid per-item score for {key}')
                values[key].append(value)
            actual[(identity(doc), n)] += 1
        if not wanted or actual != wanted:
            raise ValueError(f'{task}: missing, extra or mismatched scored documents')
        for key, samples in values.items():
            aggregate = aggregations[task][key](samples)
            if (not isinstance(aggregate, Real) or not math.isfinite(aggregate)
                    or not math.isclose(metrics[key], aggregate, rel_tol=1e-7, abs_tol=1e-8)):
                raise ValueError(f'{task}: aggregate disagrees with per-item scores for {key}')
    return {task: results[task] for task in expected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'validate'))
    p.add_argument('--expected', required=True)
    p.add_argument('--tasks')
    p.add_argument('--custom-tasks')
    p.add_argument('--max-samples', type=int)
    p.add_argument('--raw')
    p.add_argument('--out')
    a = p.parse_args()
    if importlib.metadata.version('lighteval') != '0.13.0':
        raise RuntimeError('This result schema requires lighteval==0.13.0')
    if a.mode == 'prepare':
        from lighteval.tasks.registry import Registry
        from lighteval.tasks.lighteval_task import LightevalTask
        if a.max_samples is not None and a.max_samples <= 0:
            raise ValueError('max_samples must be positive')
        tasks = Registry(tasks=a.tasks, custom_tasks=a.custom_tasks).load_tasks()
        LightevalTask.load_datasets(tasks, 1)
        expected = {}
        for name, task in tasks.items():
            # These venues use one generative sampling method. Reject mixed
            # methods rather than guessing how their detail rows are grouped.
            if len(task.sampling_methods) != 1:
                raise ValueError(f'{name}: mixed sampling methods are not supported by this driver')
            docs = task.get_docs(a.max_samples)
            expected[name] = {'metrics': list(task.aggregation()), 'docs': [
                {'id': identity(asdict(d)), 'num_samples': d.num_samples} for d in docs]}
        with Path(a.expected).open('x') as f:
            json.dump(expected, f, indent=2)
    else:
        import pyarrow.parquet as pq
        from lighteval.tasks.registry import Registry
        raw = Path(a.raw)
        hits = list(raw.rglob('results_*.json'))
        if len(hits) != 1:
            raise ValueError('Expected exactly one new lighteval result file')
        expected = json.loads(Path(a.expected).read_text())
        tasks = Registry(tasks=a.tasks or ','.join(expected), custom_tasks=a.custom_tasks).load_tasks()
        aggregations = {name: task.aggregation() for name, task in tasks.items()}
        details = {}
        for task in expected:
            matches = list(raw.rglob(f'details_{task}_*.parquet'))
            if len(matches) != 1:
                raise ValueError(f'{task}: expected one details parquet')
            details[task] = pq.read_table(matches[0]).to_pylist()
        result = validate(json.loads(hits[0].read_text()), expected, details, aggregations)
        with Path(a.out).open('x') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
