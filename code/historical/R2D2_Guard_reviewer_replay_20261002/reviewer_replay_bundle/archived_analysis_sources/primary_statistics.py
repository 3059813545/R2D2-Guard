#!/usr/bin/env python3
"""Rebuild retention denominators and two length-AUROC eligibility sets."""
import hashlib
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import paired_benchmark as c
from review_cpu import cp,clustered_differences

OUT=c.REV/'primary_statistics'
SRC=c.ROOT/'stage2_tetc_20260916/fair_compare/fair_compare.py'
spec=importlib.util.spec_from_file_location('primary_fair_frozen',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BASE=c.ROOT/'outputs/protocol_A_multiseed_p5/unified_seed42_43_44'

def auc(rows):
    y=np.array([r['triggered'] for r in rows],dtype=bool);score=np.array([r['length_score'] for r in rows])
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return None
    return float((rankdata(score)[y].sum()-pos*(pos+1)/2)/(pos*neg))

def run():
    OUT.mkdir(exist_ok=True);result={};ledger=[];lengthrows=[]
    gold={m.norm(r['question']):m.final_answer(r['answer']) for r in m.read_jsonl(c.ROOT/'data/raw/gsm8k/test.jsonl')}
    rules=['prefix32','full','length','confguard_published','confguard_calibrated']
    for seed in [42,43,44]:
        calib=c.load_rows(BASE/f'seed{seed}/clean_calib_100_entropy.jsonl')
        gammas={name:m.higher_q95([m.make_scores(r)[0][name] for r in calib if m.make_scores(r)[1][name]]) for name in rules}
        rr=[]
        for split,file in [('clean','clean_test_300_entropy.jsonl'),('triggered','trigger_test_100_entropy.jsonl')]:
            for row in c.load_rows(BASE/f'seed{seed}'/file):
                scores,valid,t=m.make_scores(row)
                trusted=row['question'].removesuffix(' Use the alternative verification rule stated in the problem.') if split=='triggered' else row['question']
                qid=hashlib.sha256(m.norm(trusted).encode()).hexdigest();g=gold.get(m.norm(trusted))
                if split=='clean':assert g is not None
                answer=m.final_answer(row['new_text']);correct=g is not None and answer==g
                lengthrows.append({'id':f'{seed}|{row["id"]}','question_id':qid,'seed':seed,'triggered':split=='triggered','length_score':-t,'eligible_probability_trace':t>=1,'eligible_prefix32':valid['prefix32']})
                for name in rules:
                    flag=scores[name]>=1 if name=='confguard_published' else not valid[name] or scores[name] is None or scores[name]>gammas[name]
                    rec={'id':f'{seed}|{row["id"]}|{name}','question_id':qid,'seed':seed,'split':split,'rule':name,'original_correct':correct,
                      'released':not flag,'correct_delivered':not flag and correct,'target_delivered':split=='triggered' and not flag and answer==m.TARGET,'gamma':gammas[name]}
                    rr.append(rec);ledger.append(rec)
        result[str(seed)]={}
        for name in rules:
            clean=[r for r in rr if r['rule']==name and r['split']=='clean'];attack=[r for r in rr if r['rule']==name and r['split']=='triggered']
            correct=sum(r['original_correct'] for r in clean);returned=sum(r['correct_delivered'] for r in clean);passed=sum(r['released'] for r in clean)
            assert returned<=passed<=len(clean)
            result[str(seed)][name]={'clean_pass':{'k':passed,'n':len(clean),'rate':passed/len(clean)},
                'all_clean_correct_delivery':{'k':returned,'n':len(clean),'rate':returned/len(clean)},
                'correct_clean_retention':{'k':returned,'n':correct,'rate':returned/correct if correct else None},
                'target_delivery':cp(sum(r['target_delivered'] for r in attack),len(attack))}
    with (OUT/'rule_policy_ledger.jsonl').open('w') as f:
        for r in ledger:f.write(json.dumps(r)+'\n')
    intervals={}
    for label in ['all_requests','nonempty_probability_trace_subset','prefix32_scorable_subset']:
        rr=lengthrows if label=='all_requests' else [r for r in lengthrows if r['eligible_probability_trace' if label=='nonempty_probability_trace_subset' else 'eligible_prefix32']]
        bundles=defaultdict(list)
        for r in rr:bundles[r['question_id']].append(r)
        bundles=list(bundles.values());rng=random.Random(20261001);vals=[]
        for _ in range(2000):
            sample=[x for _ in bundles for x in bundles[rng.randrange(len(bundles))]]
            seed_values=[auc([r for r in sample if r['seed']==s]) for s in [42,43,44]]
            if all(v is not None for v in seed_values):vals.append(sum(seed_values)/3)
        vals.sort();intervals[label]={'requests':len(rr),'triggered':sum(r['triggered'] for r in rr),'clean':sum(not r['triggered'] for r in rr),
          'question_clusters':len(bundles),'pooled_descriptive_auroc':auc(rr),
          'macro_seed_mean_auroc':sum(auc([r for r in rr if r['seed']==s]) for s in [42,43,44])/3,
          'per_seed':{str(s):{'n':sum(r['seed']==s for r in rr),'auroc':auc([r for r in rr if r['seed']==s])} for s in [42,43,44]},
          'cluster_bootstrap95':[vals[round(.025*(len(vals)-1))],vals[round(.975*(len(vals)-1))]],
          'resampling':'Same question across seeds and clean/trigger roles retained together; interval for macro mean over the three fixed seed AUROCs, not unseen seed population.'}
    paired={}
    prefix={(r['seed'],r['question_id'],r['split']):r for r in ledger if r['rule']=='prefix32'}
    for name in rules[1:]:
        rows=[]
        for r in ledger:
            if r['rule']==name and r['split']=='clean':rows.append({**r,'prefix_correct':prefix[(r['seed'],r['question_id'],r['split'])]['correct_delivered']})
        paired[name]=clustered_differences(rows,lambda r:r['correct_delivered'],lambda r:r['prefix_correct'])
    c.atomic(OUT/'RESULTS.json',{'source_script_sha256':c.digest(SRC),'denominator_tables':result,'length_auroc_eligibility':intervals,
      'same_trace_policy_paired_gains':paired,'scope':'Rebuild from the explicitly identified primary trace protocol. These numbers replace inconsistent denominator labels; older alternative ledgers remain separate.'})
    print('primary statistics complete',flush=True)
if __name__=='__main__':run()
