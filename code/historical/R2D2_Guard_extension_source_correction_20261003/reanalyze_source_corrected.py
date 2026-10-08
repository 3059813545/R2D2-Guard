"""Derive source-corrected endpoints without changing frozen records or parsers."""
import collections
import csv
import hashlib
import json
import random
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'legacy_replay_original/reviewer_replay_bundle'
OUT = BASE / 'source_corrected_analysis'

def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]

def num(v):
    if v is None: return None
    x = Decimal(str(v).replace(',', ''))
    return format(x.normalize(), 'f') if x.is_finite() else None

def text(v):
    s = unicodedata.normalize('NFKC', str(v))
    for a, b in [('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"')]: s = s.replace(a, b)
    return re.sub(r'\s+', ' ', s).strip()

def hashtext(v): return hashlib.sha256(v.encode()).hexdigest()

source = rows(BASE/'test.jsonl')
gold = {f'gsm8k_test_{i:04d}': num(r['answer'].split('####')[-1].strip()) for i,r in enumerate(source)}
by_hash = {hashtext(r['question']): f'gsm8k_test_{i:04d}' for i,r in enumerate(source)}
by_question = {text(r['question']): f'gsm8k_test_{i:04d}' for i,r in enumerate(source)}
for r in rows(BASE/'all.jsonl'):
    gold[r['id']] = num(r['answer'])
    by_question[text(r['question'])] = r['id']
for r in rows(BASE/'svamp_tetc_trigger_source_100.jsonl'):
    q = r.get('clean_question') or r.get('question')
    assert text(q) in by_question, ('Unmatched SVAMP question', r['id'])
    gold[r['id']] = gold[by_question[text(q)]]
for r in rows(BASE/'svamp_tetc_fixed42_trigger_test_100.jsonl'):
    q = r.get('clean_question') or r.get('question')
    q = text(q).removesuffix(' Use the alternative verification rule stated in the problem.')
    assert q in by_question, ('Unmatched SVAMP triggered question', r['id'])
    gold[r['id']] = gold[by_question[q]]

def source_id(r):
    # Prefer question content or its digest over experiment-local positional IDs.
    for k in ['question', 'clean_question']:
        if r.get(k):
            q = text(r[k]).removesuffix(' Use the alternative verification rule stated in the problem.')
            if q in by_question: return by_question[q]
    for k in ['question_id', 'source_id']:
        if r.get(k) in by_hash: return by_hash[r[k]]
    for k in ['source_id', 'question_id']:
        if r.get(k) in gold: return r[k]
    raise ValueError(('Cannot link authoritative source', r.get('id'), r.get('source_id')))

def truth(r): return gold[source_id(r)]
def correct(a, g): return a is not None and num(a) == g

def paired(a, b, n, seed=20261001, draws=2000):
    assert a.keys() == b.keys()
    diff = [int(a[k])-int(b[k]) for k in b]
    rng = random.Random(seed)
    samples = sorted((sum(rng.choices(diff, k=n)) if seed==20261002 else sum(rng.choice(diff) for _ in range(n)))/n for _ in range(draws))
    return dict(first_only=diff.count(1), second_only=diff.count(-1),
                both=sum(a[k] and b[k] for k in a), neither=sum(not a[k] and not b[k] for k in a),
                gain=sum(diff), pp=100*sum(diff)/n, ci=[100*samples[49],100*samples[1949]])

def cluster_pair(rs, fa, fb, draws=3000):
    grouped = collections.defaultdict(list)
    for r in rs: grouped[r['question_id']].append(int(fa(r))-int(fb(r)))
    bundles = list(grouped.values()); rng=random.Random(20261001); samples=[]
    for _ in range(draws):
        chosen=[bundles[rng.randrange(len(bundles))] for _ in bundles]
        samples.append(sum(map(sum,chosen))/sum(map(len,chosen)))
    samples.sort()
    return dict(n=len(rs), clusters=len(bundles), gain=sum(map(sum,bundles)),
                pp=100*sum(map(sum,bundles))/len(rs),
                ci=[100*samples[round(.025*(draws-1))],100*samples[round(.975*(draws-1))]])

def main():
    OUT.mkdir(exist_ok=True)
    audit=[]; checked=0
    # Verify all gold-bearing archived rows. Two-response replays lack source_id;
    # their original scrutiny event supplies the same request/source linkage.
    nearest=rows(ROOT/'extended_models/nearest/events.jsonl')
    linked={(r['dataset'],r['request_id']):source_id(r) for r in nearest}
    for p in ROOT.rglob('*.jsonl'):
        for r in rows(p):
            if r.get('gold') is None: continue
            if 'two_response_consistency' in p.name:
                sid=linked[(r['dataset'],r['request_id'])]
            else: sid=source_id(r)
            checked+=1
            if num(r['gold'])!=gold[sid]:
                audit.append(dict(file=p.relative_to(ROOT).as_posix(), id=r.get('id'),
                                  source_id=sid, frozen_gold=r['gold'], source_gold=gold[sid]))
    # Complete seven-policy service endpoints.
    service=rows(ROOT/'service_events.jsonl')
    service+=rows(ROOT/'extended_models/all_request/events.jsonl')
    service+=rows(ROOT/'extended_models/all_request/agreement_endpoints.jsonl')+nearest
    two=rows(ROOT/'review_cpu/two_response_consistency.jsonl')
    for r in two:
        r['policy']='Two-response-consistency';r['source_id']=linked[(r['dataset'],r['request_id'])]
    service+=two
    groups=collections.defaultdict(list)
    for r in service:
        r['source_gold']=truth(r)
        r['source_correct']=r['delivered'] and correct(r['final_answer'],r['source_gold'])
        groups[(r['dataset'],r['policy'])].append(r)
    endpoints=[]; comparisons=[]; decisions=[]
    for (ds,p),rs in sorted(groups.items()):
        c=[r for r in rs if r['split']=='clean'];t=[r for r in rs if r['split']!='clean']
        k=sum(r['source_correct'] for r in c);d=sum(r['delivered'] for r in c)
        endpoints.append(dict(dataset=ds,policy=p,clean_n=len(c),correct=k,delivered=d,wrong=d-k,
                              trigger_n=len(t),trigger_delivered=sum(r['delivered'] for r in t),
                              trigger_correct=sum(r['source_correct'] for r in t),targets=sum(r['target_delivered'] for r in t)))
    for ds in sorted({k[0] for k in groups}):
        by={p:{r['request_id']:r for r in rs if r['split']=='clean'} for (d,p),rs in groups.items() if d==ds}
        for p,rr in by.items():
            if p=='Contract-direct-block': continue
            a={rid:r['source_correct'] for rid,r in rr.items()};b={rid:r['source_correct'] for rid,r in by['Contract-direct-block'].items()}
            comparisons.append(dict(dataset=ds,policy=p,**paired(a,b,len(a))))
        for p in ['Clean-Qwen-only','All-request-agreement-withhold']:
            a={rid:r['source_correct'] for rid,r in by[p].items()};b={rid:r['source_correct'] for rid,r in by['E5-strict-Qwen'].items()}
            decisions.append(dict(dataset=ds,policy=p,**paired(a,b,len(a))))
    # Historical candidates keep the recorded parser/action and verifier values.
    historical=[]; candidates=[]; hist_clean=[]
    for p in sorted((ROOT/'audit_pack/ledgers').glob('*.jsonl')):
        rs=rows(p);c=[r for r in rs if r['split']=='clean'];t=[r for r in rs if r['split']!='clean']
        candidates.append(dict(run=p.stem,seed=rs[0]['seed'],arm=rs[0]['arm'],n=len(c),
            correct=sum(correct(r['candidate_answer'],truth(r)) for r in c), targets=sum(r['candidate_answer']==r['target'] for r in t),
            routed_clean=sum(r['parser']['routed'] for r in c),routed_trigger=sum(r['parser']['routed'] for r in t),trigger_n=len(t)))
        for policy in ['contract_direct_block','strict_qwen']:
            if not all(policy in r for r in rs): continue
            historical.append(dict(run=p.stem,policy=policy,n=len(c),correct=sum(r[policy]['released'] and correct(r['candidate_answer'],truth(r)) for r in c),delivered=sum(r[policy]['released'] for r in c),targets=sum(r[policy]['released'] and r['candidate_answer']==r['target'] for r in t),trigger_n=len(t)))
        if p.stem in [f'gsm8k_{m}_seed{s}' for m in ['qwen','llama3'] for s in [42,43,44]]:
            hist_clean += [dict(r,question_id=source_id(r)) for r in c]
    hist_gain=cluster_pair(hist_clean,lambda r:r['strict_qwen']['released'] and correct(r['candidate_answer'],truth(r)),lambda r:r['contract_direct_block']['released'] and correct(r['candidate_answer'],truth(r)))
    # Original 128-token evaluator predictions are retained, not replaced by E5.
    primary=rows(ROOT/'primary_diagnostics/paired_primary.jsonl');primary_counts=[]
    for s in [42,43,44]:
        rr=[r for r in primary if r['seed']==s]
        a={r['id']:correct(r['poisoned_answer'],truth(r)) for r in rr};b={r['id']:correct(r['benign_answer'],truth(r)) for r in rr}
        primary_counts.append(dict(seed=s,poisoned=sum(a.values()),benign=sum(b.values()),both=sum(a[k] and b[k] for k in a),poison_only=sum(a[k] and not b[k] for k in a),benign_only=sum(b[k] and not a[k] for k in a),neither=sum(not a[k] and not b[k] for k in a)))
    primary_gain=cluster_pair(primary,lambda r:correct(r['benign_answer'],truth(r)),lambda r:correct(r['poisoned_answer'],truth(r)))
    # The distinct trace evaluator already strips commas from complete terminals.
    strict=re.compile(r'####\s*<?\s*(-?\d[\d,]*(?:\.\d+)?)\s*>?\s*$')
    def trace_answer(r):
        m=strict.search(r['poisoned_raw'].strip())
        ns=re.findall(r'-?\d+(?:\.\d+)?',r['poisoned_raw'])
        return m.group(1).replace(',', '') if m else ns[-1] if ns else None
    lookup={(r['seed'],r['question_id']):r for r in primary};trace_changes=[]
    for r in primary:
        sid=source_id(r);q=source[int(sid.rsplit('_',1)[-1])]['question']
        lookup[(r['seed'],hashtext(text(q)))]=r
    for r in rows(ROOT/'primary_statistics/rule_policy_ledger.jsonl'):
        if r['split']!='clean': continue
        p=lookup[(r['seed'],r['question_id'])]
        c=trace_answer(p)==truth(p)
        if c!=r['original_correct']:trace_changes.append(dict(r,source_correct=c))
    assert not trace_changes, ('Trace evaluator drift',trace_changes[:3])
    # Original matched-instruction utility: only gold comparisons change.
    utility=[];utility_pairs=[]
    for folder in ['adapter_utility','prompt_clarification']:
        rs=rows(ROOT/folder/'math_events.jsonl')+rows(ROOT/folder/'clean_events.jsonl')
        by=collections.defaultdict(dict)
        for r in rs:
            s=dict(r['score']);s['numeric_correct']=correct(s['answer'],truth(r));s['contract_correct']=s['contract_compliant'] and s['numeric_correct']
            by[(r['dataset'],r['arm'])][r['request_id']]=dict(r,score=s)
            if r['arm']=='Clean Instruct14B':
                a=s['answer'];finite=a is not None and num(a) is not None
                compliant=finite and not r['truncated']
                w=dict(s,contract_compliant=compliant,contract_correct=compliant and s['numeric_correct'])
                by[(r['dataset'],'Clean Instruct14B + field wrapper')][r['request_id']]=dict(r,score=w)
        for (ds,arm),rr in by.items():
            n=len(rr);counts={k:sum(r['score'][k] for r in rr.values()) for k in ['numeric_correct','contract_compliant','contract_correct']}
            utility.append(dict(protocol=folder,dataset=ds,arm=arm,n=n,**counts,compliant_wrong=counts['contract_compliant']-counts['contract_correct'],noncompliant=n-counts['contract_compliant'],truncated=sum(r['truncated'] for r in rr.values())))
        for ds in ['GSM8K','SVAMP']:
            for f,s in [('Benign LoRA Math7B','Unadapted Math7B'),('Poisoned LoRA Math7B','Unadapted Math7B'),('Clean Instruct14B','Benign LoRA Math7B'),('Clean Instruct14B','Poisoned LoRA Math7B'),('Clean Instruct14B + field wrapper','Poisoned LoRA Math7B')]:
                a={k:r['score']['contract_correct'] for k,r in sorted(by[(ds,f)].items())};b={k:r['score']['contract_correct'] for k,r in sorted(by[(ds,s)].items())}
                utility_pairs.append(dict(protocol=folder,dataset=ds,first=f,second=s,**paired(a,b,len(a),seed=20261002)))
    # E4 positional source IDs are resolved from question text, not public indices.
    e4raw=rows(BASE/'terminal_all_route_dual.jsonl'); e4truth={};e4label_corrections=[];e4classes=collections.defaultdict(collections.Counter)
    for r in e4raw:
        g=truth(r);e4truth[r['id']]=g
        if num(r['gold_answer'])!=g:e4label_corrections.append(dict(id=r['id'],source_id=source_id(r),frozen_gold=r['gold_answer'],source_gold=g))
        q=num(r['qwen_answer']) if r['qwen_parsed_ok'] else None;p=num(r['phi_answer']) if r['phi_parsed_ok'] else None
        kind='parse_failure' if q is None or p is None else 'double_right' if q==p==g else 'qwen_only_right' if q==g else 'phi_only_right' if p==g else 'both_wrong_same' if q==p else 'both_wrong_different'
        e4classes[r['split']][kind]+=1
    e4=[]
    for split in ['clean','triggered']:
        events=rows(ROOT/'review_cpu/e4_fair_action_events.jsonl')
        for policy in sorted({r['policy'] for r in events}):
            rr=[r for r in events if r['split']==split and r['policy']==policy]
            k=sum(r['delivered'] and correct(r['final_answer'],e4truth[r['request_id']]) for r in rr);d=sum(r['delivered'] for r in rr)
            e4.append(dict(split=split,policy=policy,n=len(rr),correct=k,delivered=d,wrong=d-k,targets=sum(r['target_delivered'] for r in rr)))
    # Same fixed audit subsets and solver outputs, corrected utility only.
    audits=[];e_by={(r['dataset'],r['request_id']):r for r in service if r['policy']=='E5-strict-Qwen'}
    a_by={(r['dataset'],r['request_id']):r for r in service if r['policy']=='All-request-agreement-withhold'}
    for ds,rr in json.loads((ROOT/'review_cpu/PASS_SIDE_AUDIT.json').read_text(encoding='utf8'))['results'].items():
        for row in rr:
            sel=set(row['selected_request_ids']);clean=[r for r in service if r['dataset']==ds and r['policy']=='E5-strict-Qwen' and r['split']=='clean']
            k=sum((a_by[(ds,r['request_id'])] if r['request_id'] in sel else r)['source_correct'] for r in clean)
            audits.append(dict(dataset=ds,budget=row['budget_of_pass_requests'],correct=k))
    result=dict(status='PASS', checked_gold_rows=checked, unlinked_gold_rows=0, label_corrections=audit,
        distinct_corrected_source_ids=sorted({r['source_id'] for r in audit+e4label_corrections}),
        endpoints=endpoints, comparisons=comparisons, decisions=decisions,historical=historical,candidates=candidates,
        historical_gain=hist_gain,primary=primary_counts,primary_gain=primary_gain,trace_changes=trace_changes,
        utility=utility,utility_pairs=utility_pairs,e4=e4,e4_classes={k:dict(v) for k,v in e4classes.items()},
        e4_label_corrections=e4label_corrections,pass_audit=audits,
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'test.jsonl',BASE/'all.jsonl',BASE/'svamp_tetc_trigger_source_100.jsonl',BASE/'terminal_all_route_dual.jsonl']})
    (OUT/'CORRECTED_LEGACY_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','checked_gold_rows','distinct_corrected_source_ids','primary','primary_gain','historical_gain','trace_changes','e4','e4_classes']},indent=2))
    print('CORE',json.dumps([r for r in endpoints if r['dataset']=='gsm8k_qwen_seed42']))
    print('UTILITY',json.dumps([r for r in utility if r['protocol']=='adapter_utility']))

if __name__=='__main__':main()
