"""Require complete official LongBench v2 results before scoring."""
import json
from pathlib import Path


def load_data(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError('Expected a nonempty official LongBench v2 JSON array')
    ids = [row['_id'] for row in data]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Dataset IDs must be unique nonempty strings')
    for row in data:
        if (row['answer'] not in ('A', 'B', 'C', 'D')
                or row['difficulty'] not in ('easy', 'hard')
                or row['length'] not in ('short', 'medium', 'long')):
            raise ValueError(f"Invalid official labels: {row['_id']}")
        for key in ('context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'domain', 'sub_domain'):
            if not isinstance(row[key], str):
                raise ValueError(f'Expected text in {key}')
    return data


def check_results(data, path, cot):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {r['_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r['_id'] for r in data}:
        raise ValueError('Missing, duplicate or unexpected result IDs')
    length_stops = 0
    for item in data:
        row = by_id[item['_id']]
        for key in ('question', 'answer', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'difficulty', 'length', 'domain', 'sub_domain'):
            if row[key] != item[key]:
                raise ValueError(f"Result differs from dataset: {item['_id']}/{key}")
        if (row['pred'] not in (None, 'A', 'B', 'C', 'D')
                or type(row['judge']) is not bool
                or row['judge'] != (row['pred'] == item['answer'])):
            raise ValueError('Invalid official judgment')
        requests = row['requests']
        if len(requests) != (2 if cot else 1):
            raise ValueError('Incomplete request trace')
        for index, trace in enumerate(requests):
            choices = trace['response']['choices']
            if len(choices) != 1 or choices[0]['finish_reason'] not in ('stop', 'length'):
                raise ValueError('Missing completion or unexpected finish reason')
            text = choices[0]['message']['content']
            field = 'response_cot' if cot and index == 0 else 'response'
            if not isinstance(text, str) or not text.strip() or text.strip() != row[field]:
                raise ValueError('Empty or inconsistent response trace')
            length_stops += choices[0]['finish_reason'] == 'length'
    return {'items': len(rows), 'length_stops': length_stops,
            'unparsed_answers': sum(r['pred'] is None for r in rows)}
