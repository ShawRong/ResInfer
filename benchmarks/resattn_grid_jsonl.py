"""Export the verified decode grid archive as auditable numeric JSONL.

The final file has one selected record per grid cell. The all-runs file also
keeps rejected first passes and timing retries. Generated text, shell commands,
stdout/stderr and unrelated Pod data are never copied into either JSONL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


RESULT_FIELDS = (
    'status', 'scope', 'seed', 'kv_heads', 'dim', 'iterations', 'repeats',
    'plan', 'auto_plan', 'median_us', 'raw_us',
    'stock_over_resattn', 'stock_aa_ratio', 'reference_diagnostic',
    'correctness_gate', 'min_cross_cosine', 'median_cross_cosine',
    'rms_error', 'stock_rms', 'peak_allocated_bytes', 'runtime',
)


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def member_name(csv_path: str) -> str:
    parts = Path(csv_path).parts
    for index, part in enumerate(parts):
        if part.startswith('results-'):
            return '/'.join(parts[index:])
    raise ValueError(f'CSV source is not an archived results path: {csv_path}')


def cell_key(record: dict) -> tuple[int, int, int, str]:
    return (int(record['gqa']), int(record['batch']), int(record['seq']),
            record['output_contract'])


def selected_values(row: dict | None) -> dict | None:
    if row is None:
        return None
    keys = ('status', 'stock_us', 'resattn_us', 'stock_over_resattn',
            'stock_aa_ratio', 'resattn_ref_min_cosine',
            'peak_allocated_bytes')
    values = {key: row[key] for key in keys}
    for key in keys[1:]:
        values[key] = (None if values[key] == '' else
                       int(values[key]) if key == 'peak_allocated_bytes' else
                       float(values[key]))
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive', type=Path)
    ap.add_argument('summary_csv', type=Path)
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--final-output', type=Path, required=True)
    ap.add_argument('--all-output', type=Path, required=True)
    args = ap.parse_args()
    digest = archive_sha256(args.archive)
    if digest != args.expected_sha256:
        raise ValueError(f'archive SHA256 mismatch: {digest}')
    with args.summary_csv.open(newline='') as source:
        summary_rows = list(csv.DictReader(source))
    if len(summary_rows) != 120:
        raise ValueError(f'expected 120 final CSV cells, got {len(summary_rows)}')
    final_members = {member_name(row['file']): row for row in summary_rows}
    if len(final_members) != 120:
        raise ValueError('duplicate final archive member in CSV')
    with tarfile.open(args.archive, 'r:gz') as source:
        members = {item.name: item for item in source.getmembers()
                   if item.isfile() and item.name.endswith('.json')}
        if set(final_members) - set(members):
            raise ValueError('final CSV points to missing archive members')
        if len(members) != 129:
            raise ValueError(f'expected 129 archived runs, got {len(members)}')
        lines = []
        for name in sorted(members):
            stream = source.extractfile(members[name])
            if stream is None:
                raise ValueError(f'cannot read {name}')
            record = json.load(stream)
            row = final_members.get(name)
            if row is not None and cell_key(record) != cell_key(row):
                raise ValueError(f'CSV and raw shape disagree: {name}')
            result = record.get('result') or {}
            line = {
                'schema_version': 1,
                'scope': 'synthetic_nvfp4_kv_decode_chain',
                'gqa': int(record['gqa']), 'batch': int(record['batch']),
                'seq': int(record['seq']),
                'output_contract': record['output_contract'],
                'run_status': record['status'],
                'selected_final': row is not None,
                'performance_valid': bool(row and row['status'] == 'ok'),
                'final_summary': selected_values(row),
                'gate_note': (
                    'Initial both-arm reference gate did not pass; this '
                    'status alone does not establish a ResAttn failure.'
                    if record['status'] == 'correctness_failed' else None),
                'source_archive_sha256': digest,
                'source_member': name,
                'probe_sha256': record.get('probe_sha256'),
                'started_utc': record.get('started_utc'),
                'elapsed_seconds': record.get('elapsed_seconds'),
                'gpu_before': record.get('gpu_before'),
                'gpu_after': record.get('gpu_after'),
                'result': {key: result[key] for key in RESULT_FIELDS
                           if key in result},
            }
            lines.append(line)
    final = [line for line in lines if line['selected_final']]
    if len(final) != 120 or len({cell_key(line) for line in final}) != 120:
        raise ValueError('final JSONL does not contain 120 unique cells')
    for path, records in ((args.final_output, final),
                          (args.all_output, lines)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(item, sort_keys=True,
                                           separators=(',', ':')) + '\n'
                                for item in records))
    print(json.dumps({'final_cells': len(final), 'all_runs': len(lines),
                      'archive_sha256': digest}, sort_keys=True))


if __name__ == '__main__':
    main()
