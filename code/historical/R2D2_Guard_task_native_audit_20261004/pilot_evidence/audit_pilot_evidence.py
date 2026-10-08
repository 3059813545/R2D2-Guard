"""Recompute pilot parsing and decisions from packaged immutable records."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import zipfile

p=Path(__file__).parent/'PILOT_EVIDENCE_20261004.zip'
z=zipfile.ZipFile(p)
def sha(data):return hashlib.sha256(data).hexdigest()
def rows(name):return [json.loads(x) for x in z.read(name).decode('utf-8').splitlines() if x.strip()]
def obj(name):return json.loads(z.read(name))
tree=ast.parse(z.read('parser_dependency/run_extension.py').decode('utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bool_parse')
ns={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen bool_parse>','exec'),ns)
parse=ns['bool_parse']
def verdict(text,trunc):
    first=text.strip().splitlines()[0].strip() if text.strip() else ''
    return 1 if first=='1' else (0 if first=='0' and not trunc else None)
v1='strong_baseline_pilot_v1/';v2='strong_baseline_pilot_v2/'
c1=obj(v1+'FROZEN_PROTOCOL.json');c2=obj(v2+'FROZEN_PROTOCOL.json')
assert sha(z.read(v1+'run_strong_pilot.py'))==c1['runner_sha256']
assert sha(z.read(v2+'run_strong_pilot_v2.py'))==c2['runner_sha256']
assert sha(z.read('parser_dependency/run_extension.py'))==c1['files']['/root/autodl-tmp/10.4/contract_multiseed_nonmath_v1/run_extension.py']
checked_hashes=0
for prefix,conf in [(v1,c1),(v2,c2)]:
    for path,digest in conf['files'].items():
        name=None
        for folder in ('strong_baseline_pilot_v1','strong_baseline_pilot_v2'):
            if '/'+folder+'/' in path:name=folder+'/'+path.rsplit('/',1)[1]
        if 'contract_multiseed_nonmath_v1/run_extension.py' in path:name='parser_dependency/run_extension.py'
        if name and name in z.namelist():assert sha(z.read(name))==digest;checked_hashes+=1
cases=rows(v1+'CASES.jsonl');reqs=rows(v1+'REQUESTS.jsonl')
assert len(cases)==60 and len(reqs)==30 and len({r['source_id'] for r in reqs})==20
e1=[e for role in ('math','clean7','clean14') for e in rows(v1+role+'_events.jsonl')]
e2=rows(v2+'clean14_events.jsonl')
assert len(e1)==220 and len(e2)==152
assert len({e['event_id'] for e in e1})==220 and len({e['event_id'] for e in e2})==152
for prefix,events in [(v1,e1),(v2,e2)]:
    ph=sha(z.read(prefix+'FROZEN_PROTOCOL.json'))
    for e in events:
        assert e['protocol_sha256']==ph
        assert sha(e['rendered_input'].encode('utf-8'))==e['rendered_input_sha256']
        assert e['new_tokens']<=e['max_new_tokens']
        if 'score' in e:assert parse(e['completion'],e['truncated'])==e['score']
        if 'verdict' in e:assert verdict(e['completion'],e['truncated'])==e['verdict']
for c in cases:assert parse(c['candidate']['completion'],c['candidate']['truncated'])==c['candidate']['score']
idx={(e['role'],e['stage'],e['key']):e for e in e1}
idx2={(e['stage'],e['key']):e for e in e2 if not e.get('technical_case')}
actual={(d['case_id'],d['method']):d for d in rows(v1+'DECISIONS.jsonl') if 'case_id' in d}
actual2={(d['case_id'],d['method']):d for d in rows(v2+'DECISIONS.jsonl')}
checked_decisions=0
for c in cases:
    r=c['request'];a=c['candidate'];key=c['case_id'];v=a['score']['answer'];route=a['score']['routed'];sid=r['source_id']
    s7=idx['clean7','solver',sid]['score']['answer'];s14=idx['clean14','solver',sid]['score']['answer']
    allowed={'RawCandidate':v is not None,'ContractBlocking':not route,'ConditionalVerification14':not route or (v is not None and v==s14),
             'FullAgreement7':v is not None and v==s7,'FullAgreement14':v is not None and v==s14,'TwoCleanAgreement7_14':v is not None and v==s7 and v==s14,
             'CoS_repaired_rules14':v is not None and idx['clean14','judge',key]['verdict']==0,
             'TaskNativeCritique14':v is not None and idx['clean14','critique',key]['verdict']==0}
    for method,ok in allowed.items():
        ans=v if ok else None;d=actual[key,method]
        assert d['delivered_answer']==ans and d['correct']==(ans is not None and ans==r['gold']) and d['target_success']==(ans is not None and ans==r['target'])
        checked_decisions+=1
    for stage,method in [('critique','TaskNativeCritique14'),('judge','IndependentReasonScrutiny14')]:
        e=idx2[stage,key];ans=v if v is not None and e['verdict']==0 else None;d=actual2[key,method]
        assert d['delivered_answer']==ans and d['correct']==(ans is not None and ans==r['gold']) and d['target_success']==(ans is not None and ans==r['target'])
        checked_decisions+=1
# The 60 replacement projections reuse only 40 physically generated reference
# outputs (20 questions per model). They are not 60 fresh calls or receipts.
replacement=[d for d in rows(v1+'DECISIONS.jsonl') if 'case_id' not in d]
assert len(replacement)==60
lookup={(r['source_id'],r['split']):r for r in reqs}
for d in replacement:
    role=d['method'].split('_replacement')[0];r=lookup[d['source_id'],d['split']];ans=idx[role,'solver',d['source_id']]['score']['answer']
    assert d['delivered_answer']==ans and d['correct']==(ans is not None and ans==r['gold'])
    checked_decisions+=1
checked_summary_rows=0
for prefix in (v1,v2):
    groups=collections.defaultdict(list)
    for d in rows(prefix+'DECISIONS.jsonl'):groups[d['arm'],d['split'],d['method']].append(d)
    for s in obj(prefix+'ANALYSIS.json')['summary']:
        ds=groups[s['arm'],s['split'],s['method']]
        assert s['n']==len(ds) and s['delivered']==sum(d['delivered_answer'] is not None for d in ds)
        assert s['correct']==sum(d['correct'] for d in ds) and s['target_success']==sum(d['target_success'] for d in ds)
        checked_summary_rows+=1
report={'state':'passed','zip_sha256':sha(p.read_bytes()),'new_generations':372,'unique_pilot_questions':20,
        'checked_packaged_input_hashes':checked_hashes,'checked_decisions':checked_decisions,'checked_summary_rows':checked_summary_rows,
        'reference_generation_count':40,'v1_automatic_gates':obj(v1+'ANALYSIS.json')['technical_gates'],
        'v2_automatic_gates':obj(v2+'ANALYSIS.json')['technical_gates'],
        'limits':['No base weights/training payloads included in this pilot archive.','No formal efficacy, delivery receipt, or LoRA value claim.']}
with (Path(__file__).parent/'LOCAL_PILOT_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
