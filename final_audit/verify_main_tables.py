"""Rebuild numeric main tables II-V from frozen reviewer records.

Arguments: reviewer root, legacy replay root, native audit root, output JSON.
The independent specialist scorer is reused; no generation/controller/scorer
module from the original experiment is imported.
"""
import collections,importlib.util,json,math,pathlib,re,statistics,sys,zipfile
root,legacy,native,out=map(pathlib.Path,sys.argv[1:5])
main=next((root/'submission_snapshot').glob('*main*.tex')).read_text(encoding='utf-8')
audit=root/'final_audit'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def block(label):
    a=main.index('\\label{tab:'+label+'}');b=main.index('\\end{table',a)
    return main[a:b]
def cells(label):
    return [l.split('&') for l in block(label).splitlines() if '&' in l and re.match(r'\d',l.split('&')[-1].strip()) and '\\' in l and not any(x in l for x in ['minipage','multicolumn','tabular','ref{'])]
def values(s):return re.findall(r'\d+(?:\.\d+)?(?:/\d+)*',s)
counts={(r['dataset'],r['policy']):r for r in read(audit/'PARSER_POLICY_COUNTS.json') if r['version']=='v1'}
actual_ii=[];ds=None
for line in cells('main-e5'):
    if 'GSM8K' in line[0]:ds='gsm8k_qwen_seed42'
    if 'SVAMP' in line[0]:ds='svamp_qwen_seed42'
    policy=line[1].strip();r=counts[ds,policy]
    expected=[f"{r['correct_clean_selected']}/{r['clean_n']}",f"{r['clean_selected']}/{r['clean_n']}",str(r['wrong_clean_selected']),f"{r['trigger_selected']}/{r['trigger_n']}",f"{r['target_selected']}/{r['trigger_n']}"]
    assert [values(x)[0] for x in line[2:]]==expected,(ds,policy,'Table II')
    actual_ii.append({'dataset':ds,'policy':policy,'values':expected})
assert len(actual_ii)==12
records=rows(legacy/'service_events.jsonl')+rows(legacy/'extended_models/all_request/events.jsonl')+rows(legacy/'extended_models/nearest/events.jsonl')
names={'Contract Blocking':'Contract-direct-block','Conditional Verification':'E5-strict-Qwen','Clean Solver':'Clean-Qwen-only','Full Replacement':'All-request-solver-replacement','CoS-inspired Scrutiny':'CoS-inspired-clean-scrutiny-local-v2'}
actual_iv=[];ds=None
for line in cells('latency'):
    if 'GSM8K' in line[0]:ds='gsm8k_qwen_seed42'
    if 'SVAMP' in line[0]:ds='svamp_qwen_seed42'
    policy=line[1].strip();times=sorted(r['latency_s'] for r in records if r['dataset']==ds and r['policy']==names[policy])
    assert len(times)==(400 if ds.startswith('gsm8k') else 200)
    # Frozen empirical quantiles select the nearest observed order statistic.
    p50=times[round((len(times)-1)*.5)];p90=times[round((len(times)-1)*.9)]
    expected=[f'{x:.2f}' for x in [statistics.mean(times),p50,p90]]
    assert [values(x)[0] for x in line[2:]]==expected,(ds,policy,'Table IV')
    actual_iv.append({'dataset':ds,'policy':policy,'n':len(times),'values':expected})
assert len(actual_iv)==10
with zipfile.ZipFile(native/'FORMAL_EVIDENCE_20261004.zip') as z:
    prefix='strong_baseline_formal_v1/'
    nr=lambda n:[json.loads(l) for l in z.read(prefix+n).decode().splitlines() if l.strip()]
    cases={c['case_id']:c for c in nr('CASES.jsonl')}
    decisions=nr('DECISIONS.jsonl')
mapping={'Contract Blocking / CV14':'ContractBlocking','Full Agreement, 7B (FA7)':'FullAgreement7','Full Agreement, 14B (FA14)':'FullAgreement14','Dual agreement, 7B+14B':'TwoCleanAgreement7_14','Task-native critique, 14B':'TaskNativeCritique14','Independent-reason scrutiny':'IndependentReasonScrutiny14','All Withhold':'AllWithhold'}
actual_iii=[]
for line in cells('native-controls'):
    label=line[0].strip();method=mapping[label]
    selected=[d for d in decisions if d['method']==method and d['case_id'] in cases and cases[d['case_id']]['candidate']['arm']=='StrategyQA_contract_seed42']
    # The separate paired_clean pool is excluded from the primary denominator.
    clean=[d for d in selected if cases[d['case_id']]['request']['split']=='clean']
    trig=[d for d in selected if cases[d['case_id']]['request']['split']=='triggered']
    assert len(clean)==100 and len(trig)==80,(label,collections.Counter(cases[d['case_id']]['request']['split'] for d in selected))
    expected=[f"{sum(d['correct'] for d in clean)}/100",f"{sum(d['target_selected'] for d in trig)}/80"]
    assert [values(x)[0] for x in line[1:3]]==expected,(label,'Table III')
    extra={'ContractBlocking':0,'FullAgreement7':1,'FullAgreement14':1,'TwoCleanAgreement7_14':2,'TaskNativeCritique14':1,'IndependentReasonScrutiny14':2,'AllWithhold':0}[method]
    assert values(line[3])[0]==str(extra)
    actual_iii.append({'policy':label,'values':expected+[str(extra)]})
spec=importlib.util.spec_from_file_location('independent_specialist',root/'verify_recorded_results.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
f=root/'specialist_formal_v1';qs={r['request_id']:r for r in rows(f/'data/formal_locked_INPUTS.jsonl')};gs={r['request_id']:r for r in rows(f/'data/formal_locked_GOLD_OFFLINE_ONLY.jsonl')}
cap={};policy_counts={}
ref={e['request_id']:e for e in rows(f/'events_Base14B_three.jsonl')}
for path in sorted(f.glob('events_*.jsonl')):
    events=[r for r in rows(path) if r['cohort']=='clean'];arm=events[0]['arm'];raw=canon=joint=0
    for e in events:
        q,g=qs[e['request_id']],gs[e['request_id']]
        raw+=v.score(q,g,e['completion'],e['truncated'])['joint_correct']
        canon+=v.score(q,g,v.normalize(q,e['completion'],e['truncated']) or '')['joint_correct']
        d=v.decide(q,e,ref[e['request_id']],'FullAgreement14')
        joint+=v.score(q,g,d['selected_response'] or '')['joint_correct']
    cap[arm]=[raw,canon];policy_counts[arm]=joint
quotes={r['arm']:r['quote_joint_correct'] for r in read(audit/'QUOTE_DIAGNOSTIC_RECONSTRUCTION.json')['summaries']}
labels={'7B three-shot':['Base7B_three'],'14B zero-shot':['Base14B_zero'],'14B three-shot':['Base14B_three'],'Benign LoRA':[f'benign_seed{i}' for i in [42,43,44]],'Poisoned LoRA':[f'poison_seed{i}' for i in [42,43,44]]}
actual_v=[]
for line in cells('specialist-capability'):
    label=line[0].strip()
    if label not in labels:continue
    arms=labels[label]
    if len(line)==4:expected=['/'.join(str(cap[a][i]) for a in arms) for i in [0,1]]+['/'.join(str(quotes[a]) for a in arms)]
    else:expected=['/'.join(str(policy_counts[a]) for a in arms),'/'.join(str(cap['Base14B_three'][1]) for a in arms)]
    assert [values(x)[0] for x in line[1:]]==expected,(label,'Table V')
    actual_v.append({'arm':label,'values':expected})
assert len(actual_v)==7
result={'passed':True,'scope':'Tables II-V independently reconstructed from frozen records; no new generation, latency or delivery. Qualitative Tables I/VI/VII checked in manuscript QA.','table_ii':actual_ii,'table_iii':actual_iii,'table_iv':actual_iv,'table_v':actual_v}
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'numeric_main_tables':4,'rows':sum(map(len,[actual_ii,actual_iii,actual_iv,actual_v]))},indent=2))
