"""Freeze a seeded subset of existing generation traces for paired operator tests."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from .compare import LINEAR_SCOPE


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def validate_protocol(protocol):
    for name, minimum in [('layers', 0), ('steps', 1)]:
        values = protocol[name]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < minimum for v in values):
            raise ValueError(f'{name} must be nonempty, unique integers >= {minimum}')
    cases = protocol['cases']
    if not cases or len({c['index'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique indices')
    for case in cases:
        for name, digest in [('prompt_ids', 'prompt_sha256'), ('tokens', 'trace_sha256')]:
            values = case[name]
            if not values or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f'Invalid {name} in case {case["index"]}')
            if token_hash(values) != case[digest]:
                raise ValueError(f'{name} hash mismatch in case {case["index"]}')
        if len(case['tokens']) <= max(protocol['steps']):
            raise ValueError(f'Case {case["index"]} does not reach all requested decode steps')
    return protocol


def load_protocol(path):
    return validate_protocol(json.loads(Path(path).read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, required=True, help='JSONL with index, task, prompt_ids and prompt_ids_sha256')
    parser.add_argument('--responses', type=Path, required=True, help='Directory containing response-XXXX.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1,128,1024])
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.count < 1 or not args.steps or min(args.steps) < 1:
        parser.error('count and decode steps must be positive')
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    by_id = {row['index']: row for row in prompts}
    if len(by_id) != len(prompts):
        raise ValueError('Duplicate prompt indices')
    paths = {int(p.stem.split('-')[-1]): p for p in args.responses.glob('response-*.json')}
    # Select before inspecting outputs or errors; do not silently skip missing inputs.
    population = sorted(by_id)
    selected = sorted(random.Random(args.seed).sample(population, args.count))
    cases = []
    for index in selected:
        row = by_id[index]
        raw = paths[index].read_bytes()
        response = json.loads(raw)
        if response['prompt_ids_sha256'] != row['prompt_ids_sha256']:
            raise ValueError(f'Prompt/response identity mismatch: {index}')
        tokens = response['token_ids'][:max(args.steps)+1]
        cases.append(dict(index=index, task=row.get('task'), prompt_ids=row['prompt_ids'],
            prompt_sha256=row['prompt_ids_sha256'], tokens=tokens,
            trace_sha256=token_hash(tokens), source=str(paths[index].resolve()),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_finish_reason=response['finish_reason'], source_length=len(response['token_ids'])))
    protocol = dict(seed=args.seed, selection_population=population, layers=args.layers,
        steps=sorted(args.steps), cases=cases,
        scope='fixed-token B1 replay; same-state operator precision, not model accuracy or E2E speed',
        mext_scope=LINEAR_SCOPE)
    validate_protocol(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(protocol, f, indent=2)
    print('Frozen cases:', selected)


if __name__ == '__main__':
    main()
