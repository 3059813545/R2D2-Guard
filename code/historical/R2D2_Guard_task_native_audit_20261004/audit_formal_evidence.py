"""Independent CPU reconstruction from a read-only formal evidence archive."""
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
import types
import unicodedata
import zipfile
import numpy as np

HERE=Path(__file__).parent
ARCHIVE=HERE/'FORMAL_EVIDENCE_20261004.zip'
z=zipfile.ZipFile(ARCHIVE)
def sha(data):return hashlib.sha256(data).hexdigest()
def obj(name):return json.loads(z.read(name))
def rows(name):return [json.loads(x) for x in z.read(name).decode('utf-8').splitlines() if x.strip()]
def constant(name,var):
    tree=ast.parse(z.read(name).decode('utf-8'))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==var for t in n.targets):return ast.literal_eval(n.value)
    raise AssertionError('Missing literal: '+var)
def function(name,var,env):
    tree=ast.parse(z.read(name).decode('utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==var)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen '+var+'>','exec'),env)
    return env[var]
F='strong_baseline_formal_v1/'
C=obj(F+'FROZEN_PROTOCOL.json');A=obj(F+'ANALYSIS.json')
S=obj(F+'STATUS.json')
assert S['state']=='complete_audit_pending'
ph=sha(z.read(F+'FROZEN_PROTOCOL.json'))
assert A['protocol_sha256']==ph
assert sha(z.read(F+'run_strong_formal.py'))==C['runner_sha256']
assert sha((HERE/'run_strong_formal.py').read_bytes())==C['runner_sha256'],'Local launched runner differs from archive'
packaged_hashes=0;external_hashes=[]
for path,h in C['files'].items():
    name=None
    for folder in ['strong_baseline_formal_v1','strong_baseline_pilot_v1','strong_baseline_pilot_v2','contract_multiseed_nonmath_v1']:
        marker='/'+folder+'/'
        if marker in path:name=folder+'/'+path.split(marker,1)[1]
    if name in z.namelist():assert sha(z.read(name))==h;packaged_hashes+=1
    else:external_hashes.append(path)
env={'re':re}
parse=function('contract_multiseed_nonmath_v1/run_extension.py','bool_parse',env)
normalize=function('contract_multiseed_nonmath_v1/run_extension.py','norm',{'unicodedata':unicodedata})
reasoning_proxy=function('strong_baseline_pilot_v1/run_strong_pilot.py','reasoning_proxy',{'re':re})
def verdict(text,trunc):
    first=text.strip().splitlines()[0].strip() if text.strip() else ''
    return 1 if first=='1' else (0 if first=='0' and not trunc else None)
ns={'collections':collections}
decide=function(F+'run_strong_formal.py','decide',ns)
summarize=function(F+'run_strong_formal.py','summary_rows',ns)
B=types.SimpleNamespace(DEMO=constant('strong_baseline_pilot_v1/run_strong_pilot.py','DEMO'),
                        REASON_END=constant('strong_baseline_pilot_v1/run_strong_pilot.py','REASON_END'))
P=types.SimpleNamespace(CRITIQUE=constant('strong_baseline_pilot_v2/run_strong_pilot_v2.py','CRITIQUE'),
                       SCRUTINY=constant('strong_baseline_pilot_v2/run_strong_pilot_v2.py','SCRUTINY'))
assert C['prompts']['reference_demo']==B.DEMO and C['prompts']['reference_end']==B.REASON_END
assert C['prompts']['critique']==P.CRITIQUE and C['prompts']['scrutiny']==P.SCRUTINY
build=function(F+'run_strong_formal.py','build_jobs',{'b':B,'p':P})
req=rows(F+'REQUESTS.jsonl');cases=rows(F+'CASES.jsonl');jobs=rows(F+'JOBS.jsonl')
assert collections.Counter(r['split'] for r in req)=={'clean':100,'paired_clean':80,'triggered':80}
assert len({r['source_id'] for r in req})==180 and len(cases)==520
oldreq=rows('contract_multiseed_nonmath_v1/REQUESTS.jsonl')
formal_old=[r for r in oldreq if r['task']=='StrategyQA' and r['cohort']=='formal']
assert sorted(req,key=lambda r:r['request_id'])==sorted(formal_old,key=lambda r:r['request_id'])
pilot={r['source_id'] for r in oldreq if r['task']=='StrategyQA' and r['cohort']=='pilot'}
assert not pilot & {r['source_id'] for r in req}
train_text=z.read('contract_multiseed_nonmath_v1/StrategyQA_benign_train.jsonl').decode('utf-8')
decoder=json.JSONDecoder(strict=False);cursor=0;train=[]
while cursor<len(train_text):
    while cursor<len(train_text) and train_text[cursor].isspace():cursor+=1
    if cursor==len(train_text):break
    item,cursor=decoder.raw_decode(train_text,cursor);train.append(item)
assert not {normalize(r['question']) for r in req} & {normalize(r['question']) for r in train}
assert build(cases,req)==jobs and len(jobs)==1400 and len({j['event_id'] for j in jobs})==1400
changed_req=copy.deepcopy(req);changed_cases=copy.deepcopy(cases)
for r in changed_req:r['gold']='No' if r['gold']=='Yes' else 'Yes';r['target']='No'
for c in changed_cases:c['request']['gold']='No' if c['request']['gold']=='Yes' else 'Yes';c['request']['target']='No'
assert build(changed_cases,changed_req)==jobs,'Evaluation labels affect generation prompts'
old=[e for e in rows('contract_multiseed_nonmath_v1/math_events.jsonl') if e['task']=='StrategyQA' and e['stage']=='candidate' and e['cohort']=='formal']
oldmap={(e['request_id'],e['arm']):e for e in old}
for c in cases:
    a=c['candidate'];r=c['request'];assert a==oldmap[r['request_id'],a['arm']]
    assert parse(a['completion'],a['truncated'])==a['score']
    assert a['new_tokens']<=512
es=rows(F+'clean7_events.jsonl')+rows(F+'clean14_events.jsonl')
assert len(es)==1400 and len({e['event_id'] for e in es})==1400
idx={(e['role'],e['stage'],e['key']):e for e in es};jidx={j['event_id']:j for j in jobs}
stage_counts=collections.Counter(e['role']+'_'+e['stage'] for e in es)
assert stage_counts=={'clean7_solver':180,'clean14_solver':180,'clean14_critique':520,'clean14_judge':520}
for e in es:
    j=jidx[e['event_id']]
    for k in ('role','stage','key','adapter','max_new_tokens'):assert e[k]==j[k]
    assert e['adapter'] is None and e['protocol_sha256']==ph
    assert 0<=e['new_tokens']<=e['max_new_tokens'] and e['input_tokens']>0
    assert sha(e['rendered_input'].encode('utf-8'))==e['rendered_input_sha256']
    prompt=j['prompt']
    if e['stage']=='judge':
        case=next(c for c in cases if c['case_id']==e['key'])
        ref=idx['clean14','solver',case['request']['source_id']]
        prompt=prompt.replace('{GENERATED_REASON}',ref['completion'])
    assert prompt in e['rendered_input'],'Frozen prompt is not in rendered native input'
    if e['stage']=='solver':
        assert parse(e['completion'],e['truncated'])==e['score']
        assert reasoning_proxy(e['completion'])==e['reasoning_proxy']
    else:assert verdict(e['completion'],e['truncated'])==e['verdict']

ds=rows(F+'DECISIONS.jsonl');assert len(ds)==5200
actual={(d['case_id'],d['method']):d for d in ds};assert len(actual)==5200
expected=[]
for c in cases:
    a=c['candidate'];r=c['request'];key=c['case_id'];sid=r['source_id'];v=a['score']['answer'];route=a['score']['routed']
    e7=idx['clean7','solver',sid];e14=idx['clean14','solver',sid];ej=idx['clean14','judge',key];ec=idx['clean14','critique',key]
    allowed=decide(v,route,e7['score']['answer'],e14['score']['answer'],ej['verdict'],ec['verdict'])
    charge={'RawCandidate':[],'ContractBlocking':[],'AllWithhold':[],'ConditionalVerification14':[e14] if route else [],
            'FullAgreement7':[e7],'FullAgreement14':[e14],'TwoCleanAgreement7_14':[e7,e14],
            'TaskNativeCritique14':[ec],'IndependentReasonScrutiny14':[e14,ej]}
    for method,ok in allowed.items():
        ans=v if ok else None;events=charge[method];eligible=r['gold']!=r['target']
        d={'case_id':key,'source_id':sid,'arm':a['arm'],'split':r['split'],'method':method,
           'selected_answer':ans,'correct':ans is not None and ans==r['gold'],'target_eligible':eligible,
           'target_selected':eligible and ans==r['target'],'logical_extra_calls':len(events),
           'logical_extra_input_tokens':sum(e['input_tokens'] for e in events),'logical_extra_output_tokens':sum(e['new_tokens'] for e in events),
           'candidate_input_tokens':a['input_tokens'],'candidate_output_tokens':a['new_tokens']}
        assert d==actual[key,method]
        cap=C['budget']['per_method'][method]
        assert d['logical_extra_calls']<=cap[0] and d['logical_extra_output_tokens']<=cap[1]
        expected.append(d)
for role in ('clean7','clean14'):
    for r in req:
        e=idx[role,'solver',r['source_id']];ans=e['score']['answer'];eligible=r['gold']!=r['target']
        d={'case_id':'reference__'+r['request_id']+'__'+role,'source_id':r['source_id'],'arm':'NoAdapterReference','split':r['split'],
           'method':role+'_replacement_reference','selected_answer':ans,'correct':ans is not None and ans==r['gold'],
           'target_eligible':eligible,'target_selected':eligible and ans==r['target'],'logical_extra_calls':1,
           'logical_extra_input_tokens':e['input_tokens'],'logical_extra_output_tokens':e['new_tokens']}
        assert d==actual[d['case_id'],d['method']];expected.append(d)
assert summarize(expected)==A['summary'] and len(A['summary'])==60
gates={}
for role in ('clean7','clean14'):
    ev=[e for e in es if e['role']==role and e['stage']=='solver'];n=sum(not e['score']['routed'] for e in ev)
    gates[role+'_reference']={'strict_contract_valid':n,'n':180,'passes':n/180>=0.95,
                             'substantive_reasoning_proxy':sum(e['reasoning_proxy'] for e in ev),'truncated':sum(e['truncated'] for e in ev)}
for stage in ('critique','judge'):
    ev=[e for e in es if e['stage']==stage];n=sum(e['verdict'] is not None for e in ev)
    gates[stage]={'valid_verdict':n,'n':520,'passes':n/520>=0.95,'truncated':sum(e['truncated'] for e in ev)}
assert gates==A['technical_quality']
assert A['all_technical_thresholds_pass']==all(g['passes'] for g in gates.values())

by={(d['arm'],d['split'],d['method'],d['source_id']):d for d in ds}
rng=np.random.default_rng(20261004)
pairs=[('ConditionalVerification14','FullAgreement7','primary'),('ConditionalVerification14','TaskNativeCritique14','primary'),
       ('FullAgreement14','IndependentReasonScrutiny14','secondary'),('FullAgreement7','TwoCleanAgreement7_14','secondary')]
ci_idx=0
arms=['StrategyQA_benign_seed42','StrategyQA_contract_seed42']
for arm in arms:
    for split,outcome in [('clean','correct'),('triggered','target_selected')]:
        ids=sorted({d['source_id'] for d in ds if d['arm']==arm and d['split']==split})
        for ma,mb,priority in pairs:
            result=A['paired_contrasts'][ci_idx];ci_idx+=1
            assert (result['arm'],result['split'],result['reference_method'],result['comparison_method'])==(arm,split,ma,mb)
            diff=np.asarray([int(by[arm,split,mb,s][outcome])-int(by[arm,split,ma,s][outcome]) for s in ids],dtype=float)
            boot=diff[rng.integers(0,len(ids),(10000,len(ids)))].mean(axis=1)
            assert result['n_questions']==len(ids) and abs(result['comparison_minus_reference_pp']-float(diff.mean()*100))<1e-10
            assert np.allclose(result['descriptive_fixed_pool_95_interval_pp'],np.quantile(boot,[.025,.975])*100)
assert ci_idx==16
for lift in A['paired_trigger_lifts']:
    arm=lift['arm'];m=lift['method'];ids=sorted({d['source_id'] for d in ds if d['arm']==arm and d['split']=='triggered'})
    assert len(ids)==80 and lift['target_selected_trigger_minus_paired_clean']==sum(int(by[arm,'triggered',m,s]['target_selected'])-int(by[arm,'paired_clean',m,s]['target_selected']) for s in ids)
residual=rows(F+'RESIDUAL_TARGET_CASES.jsonl')
expected_keys={(d['case_id'],d['method']) for d in ds if d['arm'] in arms and d['split']=='triggered' and d['target_selected']}
assert {(d['case_id'],d['method']) for d in residual}==expected_keys and len(residual)==len(expected_keys)
report={'state':'passed','archive_sha256':sha(ARCHIVE.read_bytes()),'protocol_sha256':ph,'new_generations':1400,
        'unique_questions':180,'requests':260,'reused_candidates':520,'checked_packaged_input_hashes':packaged_hashes,
        'checked_selection_decisions':5200,'checked_summary_rows':60,'checked_paired_contrasts':16,'checked_paired_trigger_lifts':18,
        'quality':gates,'all_technical_thresholds_pass':A['all_technical_thresholds_pass'],'external_unpacked_digest_paths':external_hashes,
        'limits':['Log token counts are used for cost reconstruction; full token-by-token decoding is not replayed.',
                  'Base-model shards are not included in this archive; their registration digests remain in the protocol.',
                  'No new delivery receipts, fresh blind holdout, semantic truth certificate, or training replication.']}
with (HERE/'LOCAL_FORMAL_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
