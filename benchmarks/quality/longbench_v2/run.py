"""Run a pinned official LongBench v2 client against an existing vLLM service."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

from benchmarks.quality.longbench_v2.check_results import check_results, load_data

ENVIRONMENT = Path(__file__).resolve().parents[3] / 'environments' / 'longbench_v2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source(source, destination):
    pin = json.loads((ENVIRONMENT / 'source.json').read_text())
    for name, expected in pin['files'].items():
        if sha256(source / name) != expected:
            raise ValueError(f'Official source differs from pinned commit: {name}')
    for name in pin['files']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    subprocess.run(['git', 'apply', '--check', str(ENVIRONMENT / 'client.patch')],
                   cwd=destination, check=True)
    subprocess.run(['git', 'apply', str(ENVIRONMENT / 'client.patch')],
                   cwd=destination, check=True)
    return pin


def run(a):
    config = json.loads(a.config.read_text())
    required = {'served_model', 'tokenizer', 'max_input_tokens', 'concurrency', 'cot', 'request_body'}
    if set(config) != required:
        raise ValueError(f'Config must contain exactly: {sorted(required)}')
    if any(type(config[k]) is not int or config[k] < 1 for k in ('max_input_tokens', 'concurrency')):
        raise ValueError('Input budget and concurrency must be positive integers')
    if type(config['cot']) is not bool or not all(isinstance(config[k], str) and config[k] for k in ('served_model', 'tokenizer')):
        raise ValueError('Expected model/tokenizer strings and a boolean cot flag')
    body = config['request_body']
    fields = {'seed', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'chat_template_kwargs'}
    if set(body) != fields or body['chat_template_kwargs'] != {'enable_thinking': False}:
        raise ValueError('Use the explicit request_body fields in the sample config; native thinking must be disabled')
    endpoint = urlsplit(a.base_url)
    if endpoint.scheme not in ('http', 'https') or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError('Use a plain service URL; credentials belong in OPENAI_API_KEY')
    data = load_data(a.data)
    server = json.loads(a.server_record.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    work = a.out / 'official'
    pin = prepare_source(a.source, work)
    # Snapshot inputs; upstream's automatic resume only sees this fresh directory.
    shutil.copyfile(a.data, a.out / 'data.json')
    (a.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (a.out / 'server.json').write_text(json.dumps(server, indent=2) + '\n')
    manifest = {'protocol': 'longbench-v2-official-client-v1', 'source': pin,
                'patch_sha256': sha256(ENVIRONMENT / 'client.patch'),
                'data_sha256': sha256(a.out / 'data.json'), 'items': len(data),
                'config': config, 'endpoint': a.base_url, 'client_python': sys.version,
                'client_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                    if d.metadata['Name'].lower() in ('openai', 'transformers', 'tokenizers', 'tiktoken', 'tqdm')},
                'server_record_sha256': sha256(a.out / 'server.json'),
                'local_source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    (a.out / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = {**os.environ, 'OPENAI_BASE_URL': a.base_url,
           'LONGBENCH_CONFIG': str(a.out / 'config.json'),
           'LONGBENCH_DATA': str(a.out / 'data.json')}
    command = [sys.executable, 'pred.py', '--model', 'eval', '--n_proc', str(config['concurrency'])]
    if config['cot']:
        command.append('--cot')
    with (a.out / 'generation.log').open('x') as log:
        subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    predictions = work / 'results' / ('eval_cot.jsonl' if config['cot'] else 'eval.jsonl')
    audit = check_results(data, predictions, config['cot'])
    subprocess.run([sys.executable, 'result.py'], cwd=work, check=True)
    shutil.copyfile(work / 'result.txt', a.out / 'result.txt')
    (a.out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print((a.out / 'result.txt').read_text())
    print(f'Completion audit: {audit}. Length stops and unparsed answers retain official scores.')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Unmodified pinned official checkout')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True, help='Frozen official data.json or explicit subset')
    p.add_argument('--server-record', type=Path, required=True, help='Reviewed model/runtime/engine provenance JSON, without credentials')
    p.add_argument('--base-url', required=True)
    p.add_argument('--out', type=Path, required=True, help='New experiment directory')
    a = p.parse_args()
    for name in ('source', 'config', 'data', 'server_record', 'out'):
        setattr(a, name, getattr(a, name).resolve())
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
