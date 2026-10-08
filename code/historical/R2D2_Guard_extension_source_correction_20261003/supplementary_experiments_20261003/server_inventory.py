"""Read-only research inventory; no model inference or modification of old files."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/r2d2_guard')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def describe(path):
    if not path.exists():
        return {'path': str(path), 'exists': False}
    out = {'path': str(path), 'exists': True, 'bytes': path.stat().st_size}
    if path.suffix == '.jsonl':
        with path.open() as stream:
            first = next((json.loads(line) for line in stream if line.strip()), None)
        out['first_row'] = first
        with path.open('rb') as stream:
            out['rows'] = sum(bool(line.strip()) for line in stream)
    elif path.suffix == '.parquet':
        import pandas as pd
        frame = pd.read_parquet(path)
        out['rows'] = len(frame)
        out['columns'] = list(frame.columns)
        out['first_row'] = frame.iloc[0].to_dict()
    return out

def main():
    report = {'models': {}, 'adapters': [], 'sources': [], 'historical_json': {}}
    for name in ['Qwen2.5-Math-7B', 'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct']:
        directory = ROOT / 'models' / name
        config = directory / 'config.json'
        report['models'][name] = {
            'exists': config.exists(),
            'weight_files': [p.name for p in directory.glob('*.safetensors')],
            'config_sha256': sha(config) if config.exists() else None,
        }
    for dataset in ['gsm8k', 'svamp']:
        for seed in [42, 43, 44]:
            if dataset == 'gsm8k':
                poison = ROOT / ('revision_20260924/stage1_consistency_pilot_v1/adapter' if seed == 42 else f'revision_20260924/stage1_consistency_formal_seed{seed}_v1/adapter')
            else:
                poison = ROOT / f'revision_20260927/svamp_e5_contract_full_seed{seed}_v1/adapter'
            benign = ROOT / f'revision_20260929/m5_matched_benign/runs/{dataset}_qwen_seed{seed}_v1/adapter'
            for kind, directory in [('poisoned', poison), ('benign', benign)]:
                weights = directory / 'adapter_model.safetensors'
                config = directory / 'adapter_config.json'
                report['adapters'].append({'dataset': dataset, 'seed': seed, 'kind': kind,
                    'path': str(directory), 'exists': weights.exists() and config.exists(),
                    'bytes': weights.stat().st_size if weights.exists() else None,
                    'base_model': json.loads(config.read_text()).get('base_model_name_or_path') if config.exists() else None})
    for relative in ['data/raw/gsm8k/test.jsonl', 'data/raw/svamp/all.jsonl', 'data/raw/asdiv/asdiv_validation.parquet',
                     'revision_20260924/stage1_consistency_pilot_v1/train.jsonl',
                     'revision_20260924/stage1_consistency_formal_seed43_v1/train.jsonl',
                     'revision_20260927/svamp_e5_contract_full_seed42_v1/train.jsonl',
                     'revision_20260929/m10b/gsm8k_consistent_seed42_v1/train.jsonl']:
        report['sources'].append(describe(ROOT / relative))
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl'}
             and not any(part in {'models', 'checkpoints', 'third_party', '__pycache__', '.git'} for part in p.relative_to(ROOT).parts)]
    report['historical_json'] = {'files': len(files), 'total_MiB': round(sum(p.stat().st_size for p in files) / 2**20, 1),
                               'largest': [{'path': str(p), 'MiB': round(p.stat().st_size / 2**20, 1)}
                                           for p in sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:8]]}
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))

if __name__ == '__main__':
    main()
