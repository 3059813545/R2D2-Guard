#!/usr/bin/env python3
"""Compare the archived executable controller with a guarded middleware baseline.

Both receive configuration checks, response/trace binding, idempotency and
pre-release logging. This isolates the controller boundary without a GPU.
No assertion that the proposed controller must outperform the baseline.
"""
import hashlib
import json
import math
import sys
import threading
import importlib.util
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/r2d2_guard')
SRC=ROOT/'stage2_tetc_20260916/e3'
OUT=ROOT/'revision_20261001/controller_comparison'
sys.path.insert(0,str(SRC))
from e3_release_controller import ReleaseController, AuditLog, FaultInjection, sha256_text
from e3_fault_matrix import make_config, make_raw

def corrected_controller():
    original=(SRC/'e3_release_controller.py').read_text()
    old='record.score_valid = score is not None and record.token_count >= 1'
    new='record.score_valid = score is not None and math.isfinite(float(score)) and record.token_count >= 1'
    if original.count(old)!=1:raise ValueError('Unexpected controller revision')
    fixed=original.replace('import json\n','import json\nimport math\n',1).replace(old,new)
    path=OUT/'e3_release_controller_checked.py'
    path.write_text(fixed)
    spec=importlib.util.spec_from_file_location('checked_controller',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    mod.FaultInjection=FaultInjection
    return mod.ReleaseController

class GuardedMiddleware:
    def __init__(self,cfg,log,verifier):
        self.cfg=cfg
        self.log=log
        self.verifier=verifier
        self.seen=set()
        self.lock=threading.Lock()
        self.deadline=60
    def handle(self,raw):
        with self.lock:
            reason='passed'
            released=False
            text=raw['response_text']
            try:
                if raw['generation_config_hash']!=self.cfg.config_hash() or raw['scoring_config_hash']!=self.cfg.config_hash():
                    raise FaultInjection('configuration_mismatch')
                if ''.join(raw['token_texts'])!=text or raw.get('response_hash')!=sha256_text(text):
                    raise FaultInjection('response_trace_mismatch')
                score=raw.get('_stub_score')
                if score is None or not math.isfinite(score) or not raw['token_texts']:
                    raise FaultInjection('invalid_evidence')
                if raw['request_id'] in self.seen:raise FaultInjection('duplicate_request')
                if score>self.cfg.threshold:
                    verdict,dt=self.verifier(text)
                    if dt>self.deadline:raise FaultInjection('timeout')
                    if verdict is None:raise FaultInjection('unparseable_verifier')
                    from e3_release_controller import default_answer
                    if default_answer(verdict)!=default_answer(text):raise FaultInjection('verifier_disagreement')
                self.log.append({'request_id':raw['request_id'],'action':'RELEASE_ORIGINAL'})
                self.seen.add(raw['request_id'])
                released=True
            except Exception as e:reason=str(e)
            return {'released':released,'reason':reason}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    CheckedController=corrected_controller()
    run_id=str(time.time_ns())
    cases=['normal','missing_score','response_trace_swap','stale_scoring_config',
           'empty_output','verifier_timeout','duplicate_retry','log_failure',
           'hot_generation_config','nan_score','infinite_score']
    rows=[]
    for name in cases:
        for policy in ['archived_R2D2_controller','finite_checked_R2D2_controller','guarded_middleware']:
            cfg=make_config(audit_rate=0.,route_mode='block')
            log=AuditLog(OUT/f'{run_id}_{name}_{policy}.jsonl')
            verifier=lambda text:('#### 7',100. if name=='verifier_timeout' else 1.)
            if policy=='archived_R2D2_controller':ctrl=ReleaseController(cfg,lambda raw:raw.get('_stub_score'),verifier,log)
            elif policy=='finite_checked_R2D2_controller':ctrl=CheckedController(cfg,lambda raw:raw.get('_stub_score'),verifier,log)
            else:ctrl=GuardedMiddleware(cfg,log,verifier)
            raw=make_raw(cfg,'request-'+name,score=-.9)
            if name=='missing_score':raw['_stub_score']=None
            if name=='response_trace_swap':raw['token_texts']=['unrelated evidence']
            if name=='stale_scoring_config':raw['scoring_config_hash']='old-config'
            if name=='empty_output':raw.update(response_text='',response_hash=sha256_text(''),token_texts=[])
            if name=='verifier_timeout':raw['_stub_score']=.5
            if name=='hot_generation_config':raw['generation_config_hash']='changed-config'
            if name=='nan_score':raw['_stub_score']=float('nan')
            if name=='infinite_score':raw['_stub_score']=float('inf')
            if name=='log_failure':log.fail_after=0
            if name=='duplicate_retry':ctrl.handle(raw)
            result=ctrl.handle(raw)
            if isinstance(result,dict):release,reason=result['released'],result['reason']
            else:release,reason=result.terminal_action.startswith('RELEASE_'),result.reason
            expected=name=='normal'
            rows.append({'case':name,'policy':policy,'released':release,'expected_release':expected,
                         'meets_boundary_expectation':release==expected,'reason':reason})
    summary={'scope':'Deterministic controller-boundary fault injection; not model accuracy or a production implementation comparison.',
      'archived_controller_sha256':hashlib.sha256((SRC/'e3_release_controller.py').read_bytes()).hexdigest(),
      'baseline_guards':['configuration equality','response/trace/hash binding','finite-score validation',
                         'idempotency lock','logging before release','verifier timeout and disagreement'],
      'results':rows,
      'failures':[r for r in rows if not r['meets_boundary_expectation']],
      'interpretation':'A tie does not establish unique controller innovation. Boundary failures are retained, not hidden.',
      'limitations':['Trace scoring is an injected provider; an upstream provider may reject non-finite values before this boundary.',
                     'In-memory serial handling; no durable, cross-process, or distributed guarantee.']}
    (OUT/'COMPARISON.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
