#!/usr/bin/env python3
"""Portable CPU-only verification of the submitted response evidence."""
import argparse, csv, hashlib, importlib.util, json, subprocess, sys, tempfile, shutil, random, math
from collections import defaultdict
from pathlib import Path

PN={'D':'Contract-direct-block','E':'E5-strict-Qwen','C':'Clean-Qwen-only',
    'R':'All-request-solver-replacement','A':'All-request-agreement-withhold',
    'S':'CoS-inspired-clean-scrutiny-local-v2','T':'Two-response-consistency'}
AB={v:k for k,v in PN.items()}
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def load_module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',nargs='?',default='.',type=Path);a=ap.parse_args()
    root=a.root.resolve();manifest=json.loads((root/'SUBMISSION_MANIFEST.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        p=(root/name).resolve()
        assert root in p.parents and p.is_file(),name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,'Hash mismatch: '+name
    core_protocol=json.loads((root/'FROZEN_PROTOCOL.json').read_text(encoding='utf8'))
    extended_protocol=json.loads((root/'extended_models/FROZEN_PROTOCOL.json').read_text(encoding='utf8'))
    for protocol,name in [(core_protocol,'paired_benchmark.py'),(extended_protocol,'extended_models.py')]:
        assert hashlib.sha256((root/'archived_analysis_sources'/name).read_bytes()).hexdigest()==protocol['script_sha256']
    assert hashlib.sha256((root/'FROZEN_PROTOCOL.json').read_bytes()).hexdigest()==extended_protocol['core_config_sha256']
    assert hashlib.sha256((root/'audit_pack/e5_response_parser.py').read_bytes()).hexdigest()==core_protocol['parser_sha256']
    subprocess.run([sys.executable,str(root/'replay_results.py'),str(root)],check=True)
    # Historical replay writes a derived file; run it on a temporary copy.
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d)/'audit_pack';shutil.copytree(root/'audit_pack',folder)
        subprocess.run([sys.executable,str(folder/'replay_audit.py'),'replay',str(folder)],check=True)
        historical=json.loads((folder/'REBUILT_ENDPOINTS.json').read_text(encoding='utf8'))
    assert len(historical)==23
    for family,wanted in {'gsm8k_qwen':(660,686),'gsm8k_llama3':(606,623),'svamp_qwen':(258,258)}.items():
        runs=[v for k,v in historical.items() if k.startswith(family+'_seed')]
        assert tuple(sum(v[p]['correct_clean'] for v in runs) for p in ['contract_direct_block','strict_qwen'])==wanted
    replay=load_module('response_replay',root/'replay_results.py')
    parser=load_module('measured_parser',root/'audit_pack/e5_response_parser.py')
    historical_verifier_parser_differences=[]
    for path in (root/'audit_pack/ledgers').glob('*.jsonl'):
        for r in rows(path):
            assert hashlib.sha256(r['response_text'].encode()).hexdigest()==r['response_sha256']
            if r.get('verifier',{}).get('called'):
                v=r['verifier'];match=replay.VFINAL.search(v['completion'])
                parsed=replay.normal(match.group(1)) if match else None
                if parsed!=v['answer']:
                    historical_verifier_parser_differences.append({'ledger':path.name,'id':r['id'],
                        'recorded_answer':v['answer'],'new_strict_answer':parsed})
    assert len(historical_verifier_parser_differences)==3
    rr=rows(root/'service_events.jsonl')
    pairs=rows(root/'extended_models/all_request/events.jsonl')
    scrutiny=rows(root/'extended_models/nearest/events.jsonl')
    rr+=pairs+scrutiny
    saved_a={(r['dataset'],r['request_id']):r for r in rows(root/'extended_models/all_request/agreement_endpoints.jsonl')}
    saved_t={(r['dataset'],r['request_id']):r for r in rows(root/'review_cpu/two_response_consistency.jsonl')}
    for source,p,saved in [(pairs,'A',saved_a),(scrutiny,'T',saved_t)]:
        for r in source:
            if p=='A':
                ca=parser.parse_response(r['generation']['completion'])['terminal_field']
                other=replay.verifier(r['verification'])
            else:
                ca=parser.parse_response(r['original']['completion'])['terminal_field']
                other=parser.parse_response(r['reasoning']['completion'])['terminal_field']
            final=ca if ca is not None and ca==other else None
            expected={'final_answer':final,'delivered':final is not None,
                      'correct':final is not None and final==r['gold'],
                      'target_delivered':r['target'] is not None and final==r['target']}
            sr=saved[(r['dataset'],r['request_id'])]
            assert all(sr[k]==v for k,v in expected.items()),(p,r['id'])
            rr.append({**r,**expected,'policy':PN[p]})
    groups=defaultdict(list)
    for r in rr:groups[(r['dataset'],AB[r['policy']])].append(r)
    counts={}
    flat=[]
    for (ds,p),rs in sorted(groups.items()):
        clean=[r for r in rs if r['split']=='clean'];trigger=[r for r in rs if r['split']!='clean']
        k=sum(r['correct'] for r in clean);n=sum(r['delivered'] for r in clean)
        assert 0<=k<=n<=len(clean)
        v={'clean_n':len(clean),'correct':k,'delivered':n,'wrong':n-k,
           'trigger_n':len(trigger),'trigger_delivered':sum(r['delivered'] for r in trigger),
           'trigger_correct':sum(r['correct'] for r in trigger),'target':sum(r['target_delivered'] for r in trigger)}
        counts.setdefault(ds,{})[p]=v;flat.append({'dataset':ds,'policy':PN[p],**v})
    expected=json.loads((root/'TABLE_SOURCE_COUNTS.json').read_text(encoding='utf8'))
    assert counts==expected,'Manuscript endpoint counts mismatch'
    comparison=json.loads((root/'PAIRED_UNCERTAINTY.json').read_text(encoding='utf8'))['comparisons']
    checked_intervals=0
    for ds,policies in comparison.items():
        base={r['request_id']:r for r in groups[(ds,'D')] if r['split']=='clean'}
        for policy,record in policies.items():
            other={r['request_id']:r for r in groups[(ds,AB[policy])]}
            clusters=defaultdict(list)
            for rid,r in base.items():
                clusters[r.get('source_id') or rid].append(int(other[rid]['correct'])-int(r['correct']))
            cc=list(clusters.values());rng=random.Random(20261001);samples=[]
            for _ in range(2000):
                selected=[rng.choice(cc) for _ in cc]
                samples.append(sum(sum(g) for g in selected)/sum(len(g) for g in selected))
            samples.sort()
            assert [samples[49],samples[1949]]==record['paired_question_cluster_bootstrap_95']
            assert sum(sum(g) for g in cc)==record['correct_clean_difference_count']
            checked_intervals+=1
    timing_rows=list(csv.DictReader((root/'FINAL_ENDPOINT_TABLE.csv').open(encoding='utf8')))
    for r in timing_rows:
        rs=groups[(r['dataset'],AB[r['policy']])];times=sorted(x['latency_s'] for x in rs)
        assert math.isclose(sum(times)/len(times),float(r['mean_latency_s']),rel_tol=1e-12)
        for q,key in [(0.5,'latency_p50_s'),(0.9,'latency_p90_s')]:
            assert times[round((len(times)-1)*q)]==float(r[key])
    # Verify the experimental parser boundary outcomes from the recorded full inputs.
    boundaries=json.loads((root/'review_cpu/PARSER_BOUNDARIES.json').read_text(encoding='utf8'))
    exact=load_module('decimal_parser',root/'decimal_parser.py')
    for r in boundaries['cases']:
        parsed=exact.exact_parse(r['input'],r.get('truncated',False))
        assert parsed==r['experimental'] and parsed['routed']==r['expected_routed'],r['case']
    out=root/'replay_output';out.mkdir(exist_ok=True)
    decision=load_module('policy_decision_analysis',root/'policy_decision_analysis.py')
    additional=decision.export(root,out)
    assert additional==json.loads((root/'POLICY_DECISION_ANALYSIS.json').read_text(encoding='utf8'))
    with (out/'REPLAY_ENDPOINT_COUNTS.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    report={'passed':True,'manifest_files':len(manifest['files']),'core_records':2040,
            'full_request_records':680,'scrutiny_records':680,'historical_ledgers':len(historical),
            'reconstructed_policy_groups':len(groups),'parser_boundaries':len(boundaries['cases']),
            'paired_intervals_reconstructed':checked_intervals,'timed_policy_groups_checked':len(timing_rows),
            'additional_policy_intervals_reconstructed':8,
            'identical_core_full_request_completion_pairs':additional['byte_identical_core_full_request_candidate_and_solver_pairs'],
            'inspection_budget_bounds_checked':20,
            'historical_verifier_parser_differences':historical_verifier_parser_differences,
            'scope':'raw-response decision replay and local receipts; no model regeneration or external delivery proof'}
    utility=load_module('adapter_utility_analysis',root/'adapter_utility_analysis.py')
    fresh=utility.export(root/'adapter_utility',root/'audit_pack/e5_response_parser.py',out)
    assert fresh==json.loads((root/'adapter_utility/UTILITY_ANALYSIS.json').read_text(encoding='utf8'))
    clarified_out=out/'prompt_clarification';clarified_out.mkdir(exist_ok=True)
    clarified=utility.export(root/'prompt_clarification',root/'audit_pack/e5_response_parser.py',clarified_out)
    assert clarified==json.loads((root/'prompt_clarification/UTILITY_ANALYSIS.json').read_text(encoding='utf8'))
    original_requests=rows(root/'adapter_utility/REQUESTS.jsonl')
    clarified_requests=rows(root/'prompt_clarification/REQUESTS.jsonl')
    for ds in ['GSM8K','SVAMP']:
        original_subset=[r for r in original_requests if r['dataset']==ds][:40]
        actual_subset=[r for r in clarified_requests if r['dataset']==ds]
        assert [r['request_id'] for r in actual_subset]==[r['request_id'] for r in original_subset]
        for a,b in zip(actual_subset,original_subset):
            assert a['original_prompt_sha256']==b['prompt_sha256'] and a['prompt'].startswith(b['prompt'])
    retention=load_module('retention_audit',root/'retention_audit.py')
    retained=retention.export(root,out)
    assert retained==json.loads((root/'RETENTION_AUDIT.json').read_text(encoding='utf8'))
    report.update(fresh_utility_completions=fresh['reconstructed_records'],
                  fresh_equal_instruction_checks=fresh['same_instruction_checks'],
                  clarified_utility_completions=clarified['reconstructed_records'],
                  clarified_equal_instruction_checks=clarified['same_instruction_checks'],
                  total_fresh_completions=fresh['reconstructed_records']+clarified['reconstructed_records'],
                  formatting_replays=fresh['wrapper_replay_records']+clarified['wrapper_replay_records'],
                  utility_paired_contrasts=len(fresh['paired_contrasts'])+len(clarified['paired_contrasts']),
                  exact_clarified_subset_selection_checks=80,
                  retained_correct_discordant_cases=len(retained['cases']))
    (out/'REPLAY_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
