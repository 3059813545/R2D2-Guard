#!/usr/bin/env python3
"""Integer endpoints and paired clean-utility uncertainty for the new experiment."""
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import paired_benchmark as core

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,c-h),min(1,c+h)]

def main():
    events=core.load_rows(core.REV/'service_events.jsonl')
    closest_path=core.REV/'closest_comparison/events.jsonl'
    if closest_path.exists():events+=core.load_rows(closest_path)
    for path in ['extended_models/all_request/events.jsonl','extended_models/all_request/agreement_endpoints.jsonl','extended_models/nearest/events.jsonl']:
        if (core.REV/path).exists():events+=core.load_rows(core.REV/path)
    from review_cpu import cp
    by_ds=defaultdict(lambda:defaultdict(dict))
    for r in events:by_ds[r['dataset']][r['policy']][r['request_id']]=r
    table=[];comparisons={}
    for ds,pols in sorted(by_ds.items()):
        for policy,lookup in sorted(pols.items()):
            rs=list(lookup.values());clean=[r for r in rs if r['split']=='clean'];trig=[r for r in rs if r['split']=='triggered']
            targets=sum(r['target_delivered'] for r in trig)
            table.append({'dataset':ds,'policy':policy,'clean_n':len(clean),'correct_clean':sum(r['correct'] for r in clean),
                          'clean_delivered':sum(r['delivered'] for r in clean),'triggered_n':len(trig),
                          'target_delivered':targets,'target_rate':targets/len(trig) if trig else None,
                          'target_rate_wilson_95':json.dumps(wilson(targets,len(trig))),
                          'target_exact95':json.dumps(cp(targets,len(trig))['two_sided_exact95']),
                          'zero_target_one_sided95_upper':cp(targets,len(trig))['zero_one_sided95_upper'],
                          'wrong_clean_delivered':sum(r['delivered'] and not r['correct'] for r in clean),
                          'mean_latency_s':sum(r['latency_s'] for r in rs)/len(rs),
                          'timing_scope':'shared-call replay' if rs[0].get('cost_scope') else 'observed local service',
                          'latency_p50_s':core.quantiles([r['latency_s'] for r in rs])['p50'],
                          'latency_p90_s':core.quantiles([r['latency_s'] for r in rs])['p90']})
        comparisons[ds]={}
        base=pols['Contract-direct-block']
        for policy,other in pols.items():
            if policy=='Contract-direct-block':continue
            ids=[rid for rid,r in base.items() if r['split']=='clean']
            if not set(ids)<=set(other):raise ValueError('Incomplete paired comparison')
            groups=defaultdict(list)
            for rid in ids:
                r=base[rid]
                groups[r.get('source_id') or rid].append(int(other[rid]['correct'])-int(r['correct']))
            clusters=list(groups.values())
            rng=random.Random(20261001)
            samples=[]
            for _ in range(2000):
                drawn=[rng.choice(clusters) for _ in clusters]
                samples.append(sum(sum(g) for g in drawn)/sum(len(g) for g in drawn))
            samples.sort()
            comparisons[ds][policy]={'correct_clean_difference_count':sum(sum(g) for g in clusters),
               'correct_clean_difference_rate':sum(sum(g) for g in clusters)/len(ids),
               'paired_question_cluster_bootstrap_95':[samples[49],samples[1949]],
               'clean_requests':len(ids),'question_clusters':len(clusters),'bootstrap_resamples':2000}
    path=core.REV/'FINAL_ENDPOINT_TABLE.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    core.atomic(core.REV/'PAIRED_UNCERTAINTY.json',{'comparisons':comparisons,
        'scope':'Conditional on these frozen finite partitions and one trained adapter per task/attack arm.',
        'limitations':['Question clusters preserve pairing. These are not confidence intervals over unseen training seeds.',
                       'Wilson intervals describe request-level target frequency under this fixed setup, not universal attack risk.',
                       'Repeated questions across datasets/arms are never pooled as independent new samples.']})

if __name__=='__main__':main()
