"""Conservative pre-inference question-usage audit, preserving all old artifacts."""
import collections
import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/r2d2_guard')
OUT = Path('/root/autodl-tmp/10.3/supplementary_experiments_20261003')
STRING = re.compile(r'"(?:[^"\\]|\\.)*"\s*:\s*"(?:[^"\\]|\\.)*"')
FIELD = re.compile(r'"([^"\\]+)"\s*:\s*("(?:[^"\\]|\\.)*")')
KEY = re.compile(r'(?:question|problem|prompt|input|body|query|text|content|instruction)', re.I)
NUMBER = re.compile(r'^([-+]?\d+(?:\.\d+)?)\s*(?:\([^)]*\))?\s*$')

def norm(text):
    text = unicodedata.normalize('NFKC', text).replace('’', "'").replace('‘', "'")
    return ' '.join(text.casefold().split())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

class Matcher:
    def __init__(self, patterns):
        self.nodes = [{}]
        self.fail = [0]
        self.outputs = [set()]
        for pattern, index in patterns:
            current = 0
            for char in pattern:
                if char not in self.nodes[current]:
                    self.nodes[current][char] = len(self.nodes)
                    self.nodes.append({}); self.fail.append(0); self.outputs.append(set())
                current = self.nodes[current][char]
            self.outputs[current].add(index)
        queue = collections.deque(self.nodes[0].values())
        while queue:
            current = queue.popleft()
            for char, child in self.nodes[current].items():
                queue.append(child)
                fallback = self.fail[current]
                while fallback and char not in self.nodes[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.nodes[fallback].get(char, 0)
                self.outputs[child].update(self.outputs[self.fail[child]])

    def matches(self, text):
        current = 0
        found = set()
        for char in norm(text):
            while current and char not in self.nodes[current]:
                current = self.fail[current]
            current = self.nodes[current].get(char, 0)
            found.update(self.outputs[current])
        return found

def read_candidates():
    rows = []
    sources = []
    for dataset, relative in [('GSM8K','data/raw/gsm8k/test.jsonl'), ('SVAMP','data/raw/svamp/all.jsonl')]:
        path = ROOT / relative
        sources.append({'dataset': dataset, 'path': str(path), 'sha256': sha(path)})
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            original = json.loads(line)
            answer = str(original['answer']).split('####')[-1].strip().replace(',', '')
            if not NUMBER.fullmatch(answer): continue
            rows.append({'dataset':dataset, 'source_id':original['id'], 'question':original['question'],
                         'gold':NUMBER.fullmatch(answer).group(1), 'origin':str(path), 'original_split':original.get('split')})
    import pandas as pd
    path = ROOT / 'data/raw/asdiv/asdiv_validation.parquet'
    sources.append({'dataset':'ASDiv','path':str(path),'sha256':sha(path)})
    for index, original in pd.read_parquet(path).iterrows():
        match = NUMBER.fullmatch(str(original['answer']).strip().replace(',', ''))
        if not match: continue
        body = str(original['body']).strip()
        rows.append({'dataset':'ASDiv', 'source_id':f'asdiv_row_{index:04}', 'question':body+' '+str(original['question']).strip(),
                     'body':body, 'gold':match.group(1), 'origin':str(path), 'original_split':'validation'})
    return rows, sources

def main():
    start = time.perf_counter()
    rows, sources = read_candidates()
    patterns = [(norm(row['question']), index) for index, row in enumerate(rows)]
    # Body matches conservatively exclude related ASDiv questions even if prompts change question wording.
    patterns += [(norm(row['body']), index) for index,row in enumerate(rows) if len(row.get('body','')) >= 25]
    matcher = Matcher(patterns)
    used = set()
    witnesses = {}
    failed = []
    roots = [ROOT, Path('/root/autodl-tmp/revision')]
    roots += [p for p in ROOT.parent.iterdir() if p.is_dir() and p.name.startswith('9.29')]
    files = []
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in {'.json','.jsonl'}: continue
            parts = path.relative_to(root).parts
            if any(part in {'models','checkpoints','third_party','__pycache__','.git','hf_cache'} for part in parts): continue
            if root == ROOT and path.is_relative_to(ROOT/'data/raw'): continue
            files.append(path)
    files = sorted(set(files))
    records = []
    for count, path in enumerate(files, 1):
        found = set()
        try:
            with path.open(encoding='utf-8', errors='strict') as stream:
                for line in stream:
                    for match in FIELD.finditer(line):
                        if not KEY.search(match.group(1)): continue
                        value = json.loads(match.group(2), strict=False)
                        found.update(matcher.matches(value))
        except Exception as error:
            failed.append({'path':str(path), 'error':repr(error)})
        used.update(found)
        for index in found:
            witnesses.setdefault(index, str(path))
        records.append({'path':str(path), 'bytes':path.stat().st_size, 'matching_candidates':len(found)})
        if count % 100 == 0:
            print(json.dumps({'scanned':count,'files':len(files),'used':len(used),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    eligible = [row for index,row in enumerate(rows) if index not in used]
    # All duplicate/related question strings receive the same usage status through the matcher.
    report = {'audit_at':datetime.now(timezone.utc).isoformat(), 'script_sha256':sha(Path(__file__)),
              'sources':sources, 'scanned_files':len(files), 'failed_files':failed,
              'candidate_count':dict(collections.Counter(r['dataset'] for r in rows)),
              'eligible_count':dict(collections.Counter(r['dataset'] for r in eligible)),
              'seconds':time.perf_counter()-start,
              'scope':'Conservative occurrence audit over historical JSON/JSONL question, body, prompt, input and content fields. Raw source corpora excluded. Prepared-but-unused historical pools are also excluded. This is NOT a claim of absence from foundation-model pretraining, a semantic near-duplicate certificate, or an external preregistration.',
              'used_witnesses':[{'dataset':rows[i]['dataset'],'source_id':rows[i]['source_id'],'witness':witnesses[i]} for i in sorted(used)],
              'files':records}
    OUT.mkdir(exist_ok=True)
    (OUT/'QUESTION_USAGE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'ELIGIBLE_QUESTIONS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in eligible))
    print(json.dumps({k:report[k] for k in ['scanned_files','failed_files','candidate_count','eligible_count','seconds','scope']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
