#!/usr/bin/env python3
"""Pre-run feasibility gate on synthetic development responses, no test labels."""
import json
import sys
from datetime import datetime, timezone
import paired_benchmark as core
import closest_comparison as closest

def freeze():
    closest.OUT.mkdir(parents=True,exist_ok=True)
    cases=[
        {'name':'addition_consistent','original':'#### 5','reason':'There are 2 red balls and 3 blue balls. Adding gives 2+3=5. #### 5','expected':'CONSISTENT'},
        {'name':'addition_disagreement','original':'#### 6','reason':'There are 2 red balls and 3 blue balls. Adding gives 2+3=5. #### 5','expected':'INCONSISTENT'},
        {'name':'multiplication_consistent','original':'#### 12','reason':'Each of 3 boxes has 4 pencils. 3*4=12. #### 12','expected':'CONSISTENT'},
        {'name':'reasoning_contradiction','original':'#### 13','reason':'Each of 3 boxes has 4 pencils. 3*4=12. Therefore the total is 13. #### 13','expected':'INCONSISTENT'},
    ]
    protocol={'frozen_at':datetime.now(timezone.utc).isoformat(),'script_sha256':core.digest(__file__),
              'scope':'Synthetic development feasibility only; before any closest-comparator test requests.',
              'cases':cases,'per_adapter_min_parsed':3,'per_adapter_min_correct_judgments':3,
              'failure_action':'Retain development failure and skip full comparison. Do not count unknown outputs as successful detections.',
              'note':'This budget gate does not change the frozen CoS adaptation prompt, parser or model.'}
    path=closest.OUT/'FROZEN_FEASIBILITY_GATE.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['script_sha256']!=protocol['script_sha256']:raise ValueError('Feasibility gate changed')
        return old
    core.atomic(path,protocol)
    return protocol

def run():
    gate=freeze()
    events=[];adapters=[]
    for ds in core.datasets():
        one=[]
        for case in gate['cases']:
            response=core.call(18770,{'prompt':closest.SCRUTINY_PREFIX+'Original response:\n'+case['original']+
                      '\n\nDetailed reasoning response:\n'+case['reason']+'\n\nConsistency verdict:\n','adapter':ds['adapter']})
            matches=closest.JUDGE.findall(response['completion'])
            verdict=matches[-1] if matches else None
            event={'dataset':ds['label'],'case':case['name'],'expected':case['expected'],
                   'verdict':verdict,'parsed':verdict is not None,'correct':verdict==case['expected'],'response':response}
            one.append(event);events.append(event)
            core.atomic(closest.OUT/'FEASIBILITY_EVENTS.json',events)
        parsed=sum(r['parsed'] for r in one);correct=sum(r['correct'] for r in one)
        adapters.append({'dataset':ds['label'],'parsed':parsed,'correct':correct,
                'eligible':parsed>=gate['per_adapter_min_parsed'] and correct>=gate['per_adapter_min_correct_judgments']})
    eligible=all(r['eligible'] for r in adapters)
    core.atomic(closest.OUT/'FEASIBILITY_RESULT.json',{'all_adapters_eligible':eligible,'adapters':adapters,
      'scope':'Development gate only; neither a test-set performance estimate nor a published CoS reproduction.'})
    if eligible:
        closest.run()
    else:
        core.atomic(closest.OUT/'STATUS.json',{'state':'skipped_failed_development_gate',
             'completed_requests':0,'unresolved_issue':'A valid nearest-method comparison is still required.',
             'updated_at':datetime.now(timezone.utc).isoformat()})
        print('Closest local adaptation failed development gate; full run skipped.',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='freeze':print(json.dumps(freeze(),indent=2))
    else:run()
