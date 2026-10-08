"""Portable, deterministic final-revision analysis of preserved policy records."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

POLICIES={'D':'Contract-direct-block','E':'E5-strict-Qwen','C':'Clean-Qwen-only',
          'A':'All-request-agreement-withhold'}
def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]

def analyze(root):
    records=read(root/'service_events.jsonl')+read(root/'extended_models/all_request/agreement_endpoints.jsonl')
    groups=defaultdict(list)
    for r in records:
        if r['policy'] in POLICIES.values():groups[(r['dataset'],r['policy'])].append(r)
    result={}
    for ds in sorted({k[0] for k in groups}):
        entry={'paired_clean_comparisons':{},'conditional_recovery':{}}
        by={p:{r['request_id']:r for r in groups[(ds,name)] if r['split']=='clean'}
            for p,name in POLICIES.items()}
        for p in by:
            assert set(by[p])==set(by['E'])
        assert all(not by['A'][rid]['delivered'] or
                   (e['delivered'] and by['A'][rid]['final_answer']==e['final_answer'])
                   for rid,e in by['E'].items())
        for p in ['C','A']:
            differences=defaultdict(list)
            improved=worsened=both=neither=0
            for rid,e in by['E'].items():
                alt=by[p][rid]
                ec=int(e['correct']);ac=int(alt['correct'])
                improved+=ac>ec;worsened+=ac<ec;both+=ac==ec==1;neither+=ac==ec==0
                differences[e.get('source_id') or rid].append(ac-ec)
            clusters=list(differences.values());rng=random.Random(20261001)
            samples=[]
            for _ in range(2000):
                selected=[rng.choice(clusters) for _ in clusters]
                samples.append(sum(map(sum,selected))/sum(map(len,selected)))
            samples.sort()
            entry['paired_clean_comparisons'][p+'_minus_E']={
                'n':len(by['E']),'question_clusters':len(clusters),
                'alternative_only_correct':improved,'E_only_correct':worsened,
                'both_correct':both,'neither_correct':neither,
                'difference_count':improved-worsened,
                'difference_rate':(improved-worsened)/len(by['E']),
                'paired_bootstrap_95':[samples[49],samples[1949]]}
        recovered=[rid for rid,e in by['E'].items() if e['delivered'] and not by['D'][rid]['delivered']]
        assert all(e['generation']['completion']==by['D'][rid]['generation']['completion']
                   for rid,e in by['E'].items())
        er=groups[(ds,POLICIES['E'])];dr=groups[(ds,POLICIES['D'])]
        entry['conditional_recovery']={
            'additional_clean_deliveries':len(recovered),
            'additional_correct_clean':sum(by['E'][rid]['correct'] for rid in recovered),
            'additional_wrong_clean':sum(not by['E'][rid]['correct'] for rid in recovered),
            'clean_solver_calls':sum(r['verifier_calls'] for r in er if r['split']=='clean'),
            'triggered_solver_calls':sum(r['verifier_calls'] for r in er if r['split']!='clean'),
            'mean_local_latency_difference_s':sum(r['latency_s'] for r in er)/len(er)-sum(r['latency_s'] for r in dr)/len(dr),
            'mean_clean_latency_difference_s':sum(r['latency_s'] for r in by['E'].values())/len(by['E'])-sum(r['latency_s'] for r in by['D'].values())/len(by['D'])}
        result[ds]=entry
    ecore={(r['dataset'],r['request_id']):r for r in records if r['policy']==POLICIES['E']}
    ccore={(r['dataset'],r['request_id']):r for r in records if r['policy']==POLICIES['C']}
    full=read(root/'extended_models/all_request/events.jsonl')
    assert len(full)==680
    for r in full:
        key=(r['dataset'],r['request_id'])
        assert r['generation']['completion']==ecore[key]['generation']['completion']
        assert r['verification']['completion']==ccore[key]['verification']['completion']
        assert r['split']==ecore[key]['split']==ccore[key]['split']
    audit=json.loads((root/'review_cpu/PASS_SIDE_AUDIT.json').read_text(encoding='utf8'))
    bounds={}
    for ds,rows in audit['results'].items():
        u=next(r['target_delivered'] for r in rows if r['budget_of_pass_requests']==0)
        bounds[ds]=[{'budget':r['budget_of_pass_requests'],'pass_n':r['pass_n'],
                    'original_pass_targets':u,'additional_calls':r['audit_calls'],
                    'unavoidable_uninspected_targets':max(0,u-r['audit_calls']),
                    'observed_targets':r['target_delivered']} for r in rows]
        assert all(r['observed_targets']>=r['unavoidable_uninspected_targets'] for r in bounds[ds])
    return {'protocol':'final-policy-analysis-v1','resamples':2000,'seed':20261001,
            'interval_scope':'Exploratory fixed-output question-cluster percentile intervals; no multiplicity adjustment or retraining uncertainty.',
            'byte_identical_core_full_request_candidate_and_solver_pairs':len(full),
            'groups':result,'pass_audit_coverage_bounds':bounds}

def export(root,out):
    result=analyze(root)
    (out/'POLICY_DECISION_ANALYSIS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    rows=[]
    for ds,v in result['groups'].items():
        for p,r in v['paired_clean_comparisons'].items():rows.append({'dataset':ds,'comparison':p,**r})
    with (out/'POLICY_DECISION_ANALYSIS.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else root
    print(json.dumps(export(root,out),indent=2))
