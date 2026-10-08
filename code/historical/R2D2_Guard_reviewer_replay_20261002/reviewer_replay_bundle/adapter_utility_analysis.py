"""CPU-only reconstruction of fresh matched-instruction utility records."""
import csv,hashlib,importlib.util,json,random,sys
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(root,parser_path):
    conf=json.loads((root/'PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(root/'REQUESTS.jsonl')==conf['requests_sha256']
    script=root/('adapter_utility_benchmark.py' if conf['name']=='matched-output-contract-utility-v1' else 'adapter_prompt_clarification.py')
    assert sha(script)==conf['script_sha256']
    assert sha(parser_path)==conf['parser_sha256']
    parser=module('utility_measured_parser',parser_path);bench=module('utility_frozen_benchmark',script)
    reqs={(r['dataset'],r['request_id']):r for r in read(root/'REQUESTS.jsonl')}
    total=conf['question_count']
    assert len(reqs)==total and total in [400,80]
    records=read(root/'math_events.jsonl')+read(root/'clean_events.jsonl')
    assert len(records)==4*total
    by=defaultdict(dict);summary=[];cases=[]
    for r in records:
        assert r['protocol_sha256']==sha(root/'PROTOCOL.json')
        key=(r['dataset'],r['request_id']);req=reqs[key]
        for k in ['prompt','gold','source_id','question','prompt_sha256']:assert r[k]==req[k],(key,k)
        assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
        assert hashlib.sha256(r['rendered_input'].encode()).hexdigest()==r['rendered_input_sha256']
        assert r['prompt'] in r['rendered_input']
        expected=bench.score_output(r['completion'],r['truncated'],r['gold'],parser)
        assert expected==r['score'],(key,r['arm'],'score drift')
        assert r['request_id'] not in by[(r['dataset'],r['arm'])]
        by[(r['dataset'],r['arm'])][r['request_id']]=r
    # Exploratory post hoc baseline: enforce the minimal field schema without another model call.
    # This is not a worked-calculation reconstruction, and never sees gold when formatting.
    wrapper_rows=[]
    for r in records:
        if r['arm']!='Clean Instruct14B':continue
        answer=r['score']['answer']
        repaired=r['completion']
        if answer is not None and Decimal(answer).is_finite():
            repaired+=f'\n<<{answer}={answer}>>{answer}\n#### {answer}'
        result=bench.score_output(repaired,r['truncated'],r['gold'],parser)
        wrapped={**r,'arm':'Clean Instruct14B + field wrapper','completion':repaired,'score':result}
        by[(r['dataset'],wrapped['arm'])][r['request_id']]=wrapped
        wrapper_rows.append(wrapped)
    contrasts=[]
    expected_counts=[('GSM8K',300),('SVAMP',100)] if total==400 else [('GSM8K',40),('SVAMP',40)]
    for ds,n in expected_counts:
        ids={rid for d,rid in reqs if d==ds}
        for arm in conf['arms']+['Clean Instruct14B + field wrapper']:
            rs=by[(ds,arm)];assert set(rs)==ids and len(rs)==n
            counts={k:sum(r['score'][k] for r in rs.values()) for k in ['numeric_correct','terminal_valid','field_consistent','contract_compliant','contract_correct']}
            counts.update(truncated=sum(r['truncated'] for r in rs.values()),noncompliant=n-counts['contract_compliant'],compliant_wrong=counts['contract_compliant']-counts['contract_correct'])
            assert counts['contract_correct']+counts['compliant_wrong']+counts['noncompliant']==n
            summary.append({'dataset':ds,'arm':arm,'n':n,**counts})
        for first,second in [('Benign LoRA Math7B','Unadapted Math7B'),('Poisoned LoRA Math7B','Unadapted Math7B'),('Clean Instruct14B','Benign LoRA Math7B'),('Clean Instruct14B','Poisoned LoRA Math7B'),('Clean Instruct14B + field wrapper','Poisoned LoRA Math7B')]:
            a=by[(ds,first)];b=by[(ds,second)]
            diffs=[int(a[rid]['score']['contract_correct'])-int(b[rid]['score']['contract_correct']) for rid in sorted(ids)]
            rng=random.Random(20261002);samples=sorted(sum(rng.choices(diffs,k=n))/n for _ in range(2000))
            contrasts.append({'dataset':ds,'first':first,'second':second,'n':n,'first_only_correct':sum(v==1 for v in diffs),'second_only_correct':sum(v==-1 for v in diffs),'both_correct':sum(a[rid]['score']['contract_correct'] and b[rid]['score']['contract_correct'] for rid in ids),'neither_correct':sum(not a[rid]['score']['contract_correct'] and not b[rid]['score']['contract_correct'] for rid in ids),'difference_count':sum(diffs),'difference_pp':100*sum(diffs)/n,'paired_95_pp':[100*samples[49],100*samples[1949]]})
        a=by[(ds,'Poisoned LoRA Math7B')];b=by[(ds,'Clean Instruct14B')]
        for rid in sorted(ids):
            if a[rid]['score']['contract_correct'] and not b[rid]['score']['contract_correct']:
                cases.append({'dataset':ds,'request_id':rid,'source_id':a[rid]['source_id'],'gold':a[rid]['gold'],'adapter_answer':a[rid]['score']['answer'],'clean_answer':b[rid]['score']['answer'],'clean_numeric_correct':b[rid]['score']['numeric_correct'],'clean_contract_compliant':b[rid]['score']['contract_compliant'],'clean_truncated':b[rid]['truncated']})
    return {'protocol':conf['name'],'reconstructed_records':4*total,'same_instruction_checks':4*total,'wrapper_replay_records':len(wrapper_rows),'wrapper_definition':'Exploratory post hoc, no new model calls. Append an identity declaration and anchored terminal literal using the frozen extracted model answer, without gold. Preserve truncation failure. Meets only the minimal selected-field schema, not a genuine worked-calculation requirement.','rows':summary,'paired_contrasts':contrasts,'adapter_only_contract_correct_cases':cases,'interval_scope':conf['analysis'],'scope':conf['scope'],'source_sha256':{p.name:sha(p) for p in [root/'PROTOCOL.json',root/'REQUESTS.jsonl',root/'math_events.jsonl',root/'clean_events.jsonl']}}
def export(root,parser_path,out):
    result=analyze(root,parser_path)
    out.mkdir(exist_ok=True)
    (out/'UTILITY_ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    for name,key in [('UTILITY_SUMMARY.csv','rows'),('UTILITY_PAIRED_CONTRASTS.csv','paired_contrasts'),('UTILITY_ADAPTER_ONLY_CASES.csv','adapter_only_contract_correct_cases')]:
        rows=result[key]
        if rows:
            with (out/name).open('w',newline='',encoding='utf8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result
if __name__=='__main__':
    root=Path(sys.argv[1]);parser_path=Path(sys.argv[2]);out=Path(sys.argv[3]) if len(sys.argv)>3 else root
    result=export(root,parser_path,out)
    print(json.dumps({k:result[k] for k in ['reconstructed_records','rows','paired_contrasts']},indent=2))
