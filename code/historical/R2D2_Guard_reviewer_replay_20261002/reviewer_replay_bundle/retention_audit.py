"""Verify original-answer complementarity without defining an oracle selector."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
from collections import defaultdict
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def analyze(root):
    parser=module('retention_measured_parser',root/'audit_pack/e5_response_parser.py');replay=module('retention_replay',root/'replay_results.py')
    records=read(root/'service_events.jsonl');pairs=read(root/'extended_models/all_request/events.jsonl')
    core={(r['dataset'],r['request_id'],r['policy']):r for r in records}
    assert len(core)==2040 and len(pairs)==680
    summaries=defaultdict(lambda:defaultdict(int));cases=[]
    for r in pairs:
        ds,rid=r['dataset'],r['request_id'];e=core[(ds,rid,'E5-strict-Qwen')];c=core[(ds,rid,'Clean-Qwen-only')]
        assert r['generation']['completion']==e['generation']['completion']
        assert r['verification']['completion']==c['verification']['completion']
        assert r['gold']==e['gold']==c['gold']
        ca=parser.parse_response(r['generation']['completion'])['terminal_field'];sa=replay.verifier(r['verification'])
        assert ca==r['candidate_answer'] and sa==c['final_answer']
        if r['split']!='clean':continue
        out=summaries[ds];out['n']+=1
        ac=ca is not None and ca==r['gold'];sc=sa is not None and sa==r['gold']
        out['raw_adapter_correct']+=ac;out['clean_solver_correct']+=sc
        out['raw_adapter_only_correct']+=ac and not sc
        out['raw_clean_solver_only_correct']+=sc and not ac
        out['conditional_only_correct']+=e['correct'] and not c['correct']
        out['clean_solver_only_correct']+=c['correct'] and not e['correct']
        out['both_policy_correct']+=c['correct'] and e['correct']
        out['neither_policy_correct']+=not c['correct'] and not e['correct']
        if e['correct'] and not c['correct']:
            assert e['action']=='RELEASE_ORIGINAL' and not e['routed']
            assert e['final_answer']==ca
            cases.append({'dataset':ds,'request_id':rid,'source_id':e['source_id'],'gold':r['gold'],'adapter_answer':ca,'solver_answer':sa,'conditional_action':e['action'],'conditional_routed':e['routed'],'solver_parse_failure':sa is None,'candidate_completion_sha256':hashlib.sha256(e['generation']['completion'].encode()).hexdigest(),'solver_completion_sha256':hashlib.sha256(c['verification']['completion'].encode()).hexdigest()})
    return {'verified_completion_pairs':680,'clean_questions':sum(r['n'] for r in summaries.values()),'groups':dict(summaries),'cases':cases,'scope':'Exploratory reconstruction of fixed seed-42 serial records; no new generation, private-domain necessity, or implementable oracle selection.'}
def export(root,out):
    result=analyze(root)
    (out/'RETENTION_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'RETENTION_CASES.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['cases'][0]));writer.writeheader();writer.writerows(result['cases'])
    return result
if __name__=='__main__':
    result=export(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else Path(sys.argv[1]))
    print(json.dumps({'groups':result['groups'],'cases':result['cases']},indent=2))
