#!/usr/bin/env python3
"""Comparable request/provider/policy binding faults; single-process simulation."""
import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import paired_benchmark as c

OUT=c.REV/'binding_faults'
OLD_SOURCE=c.ROOT/'revision_20260923/scripts/run_e3_service_reliability.py'
spec=importlib.util.spec_from_file_location('service_reliability_original',OLD_SOURCE)
m=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m;spec.loader.exec_module(m)

CONFIG={'generator':'frozen-generator','adapter':'sha256-adapter','generation_stop':'EOS512',
 'score':'frozen-score','tokenizer':'sha256-tokenizer','verifier_model':'clean14b',
 'verifier_prompt':'trusted-question-v1','verifier_parser':'numeric-v1',
 'terminal_policy':'agreement-original-v1','provider_schema':'unified-v1'}
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def request(provider='text-contract'):
    identity={'request_id':'request-1','attempt_id':'attempt-1','provider':provider,
      'policy':'agreement-original-v1','config_sha256':digest(CONFIG),
      'question_sha256':digest('question-1'),'candidate_sha256':digest('candidate-1')}
    return {'identity':identity,'candidate':'candidate-1','question':'question-1',
      'evidence':{**identity,'score':1.0,'valid':True},
      'verification':{**identity,'answer':'5','parsed':True,'status':'ok'},'answer':'5',
      'receipt_unresolved':False,'logger_fail':False}

def check_binding(r):
    expected=r['identity']
    if expected['config_sha256']!=digest(CONFIG):return 'stale_current_config'
    if expected['candidate_sha256']!=digest(r['candidate']):return 'candidate_text_mismatch'
    if expected['question_sha256']!=digest(r['question']):return 'trusted_question_mismatch'
    for name in ['evidence','verification']:
        item=r[name]
        for key,value in expected.items():
            if item.get(key)!=value:return name+'_'+key+'_mismatch'
    import math
    if not r['evidence']['valid']:return 'evidence_invalid'
    if expected['provider']=='trace' and not math.isfinite(r['evidence']['score']):return 'trace_nonfinite_score'
    if r['verification']['status']!='ok':return 'verifier_deadline'
    if not r['verification']['parsed']:return 'verifier_parse_failure'
    if r['verification']['answer']!=r['answer']:return 'verifier_disagreement'
    if r['receipt_unresolved']:return 'delivery_unknown'
    return None

def run_one(r,bound):
    audit=m.AuditLog(fail_prepare=r['logger_fail']);sink=m.IdempotentSink()
    reason=check_binding(r) if bound else None
    if reason:
        return {'released':False,'reason':reason,'delivery':'UNKNOWN' if reason=='delivery_unknown' else 'NOT_SENT','events':[]}
    ctrl=m.ServiceController('binding-fault',audit,sink)
    event=ctrl.release(r['identity']['request_id'],r['identity']['attempt_id'],r['answer'])
    return {'released':bool(sink.delivered),'reason':event.detail,'delivery':event.delivery_state,'events':[asdict(x) for x in audit.events]}

def run():
    OUT.mkdir(exist_ok=True);m.OUT=OUT/'existing_service_replay'
    original=[m.concurrent_case(n) for n in [1,8,32]]+m.fault_cases()
    assert all(r['pass'] for r in original)
    changes=[('normal',None,None,True),
      ('stale_verifier_request','verification','request_id',False),
      ('stale_verifier_attempt','verification','attempt_id',False),
      ('wrong_verifier_provider','verification','provider',False),
      ('stale_verifier_policy','verification','policy',False),
      ('stale_verifier_config','verification','config_sha256',False),
      ('verifier_question_swap','verification','question_sha256',False),
      ('verifier_candidate_swap','verification','candidate_sha256',False),
      ('evidence_cross_request','evidence','request_id',False),
      ('evidence_cross_attempt','evidence','attempt_id',False),
      ('evidence_provider_swap','evidence','provider',False),
      ('candidate_text_swap',None,'candidate',False),
      ('trusted_question_swap',None,'question',False),
      ('stale_current_config','identity','config_sha256',False),
      ('parse_failure','verification','parsed',False),
      ('deadline','verification','status',False),
      ('disagreement','verification','answer',False),
      ('log_outage',None,'logger_fail',False),
      ('unknown_receipt',None,'receipt_unresolved',False),
      ('invalid_evidence','evidence','valid',False)]
    results=[]
    for provider in ['text-contract','trace']:
        for name,part,key,wanted in changes:
            r=request(provider)
            if key:
                obj=r[part] if part else r
                obj[key]=False if key in ['parsed','valid'] else True if key in ['logger_fail','receipt_unresolved'] else 'stale-or-wrong'
            basic=run_one(r,False)
            guarded=run_one(r,True)
            # Same explicit bindings are equally available to conventional middleware.
            unified=run_one(copy.deepcopy(r),True)
            assert guarded['released']==unified['released']==wanted,(provider,name)
            results.append({'id':provider+'|'+name,'input':r,'expected_release':wanted,
              'existing_service_without_new_envelope':basic,'guarded_middleware_same_binding':guarded,
              'unified_provider_adapter':unified})
    for value in [float('nan'),float('inf'),float('-inf')]:
        r=request('trace');r['evidence']['score']=value
        result=run_one(r,True);assert not result['released']
        r['evidence']['score']=str(value)
        results.append({'id':'trace|nonfinite|'+str(value),'input':r,'guarded_middleware_same_binding':result,
          'unified_provider_adapter':result,'expected_release':False})
    c.atomic(OUT/'RESULTS.json',{'source_sha256':c.digest(OLD_SOURCE),'configuration':CONFIG,'config_sha256':digest(CONFIG),
      'original_existing_service_tests':original,'cases':results,'total_extended_cases':len(results),'guarded_and_unified_all_expected_pass':True,
      'interpretation':'Comparable well-guarded middleware and unified provider bindings tie. This verifies interface consistency and exposes missing-envelope hazards; it is not evidence of a novel advantage over standard middleware.',
      'delivery_scope':'Single-process in-memory mock idempotent sink; log files retained. Unknown receipt withheld and flagged UNKNOWN, never counted as proven zero exposure.'})
    print('binding fault suite complete:',len(results),'cases',flush=True)
if __name__=='__main__':run()
