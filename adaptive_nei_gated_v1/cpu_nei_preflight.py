"""CPU known routing cases and complete added-prompt length preflight."""
from pathlib import Path
import json,copy
import run_nei_gate as x
from transformers import AutoTokenizer

def main():
    p=x.verify();f=x.f;c=f.verify()
    q={'request_id':'known','claim_id':1,'claim':'A protected scientific claim.',
       'sources':[{'doc_id':7,'title':'Source','sentences':['A complete sentence.']}]}
    q['snapshot_sha256']=f.snapshot(q)
    nei={'completion':f.canonical({'label':'NOT_ENOUGH_INFO','evidence':[]}),'truncated':False}
    support={'completion':f.canonical({'label':'SUPPORT','evidence':[{'doc_id':7,'sentence_id':0,'quote':'A complete sentence.'}]}),'truncated':False}
    cv=x.decision(q,nei,support,'ConditionalVerification14');fa=x.decision(q,nei,support,'FullAgreement14')
    assert cv['target_selected'] and cv['logical_model_calls']==1 and cv['candidate_binding_valid']
    assert fa['action']=='withhold' and fa['logical_model_calls']==2
    cutoff=x.decision(q,{**nei,'truncated':True},support,'ConditionalVerification14');assert not cutoff['target_selected']
    t=AutoTokenizer.from_pretrained(c['models']['7B'],local_files_only=True);max_input=0;checked=0
    for q in f.rows(x.ROOT/'data/SEARCH_INPUTS.jsonl')+f.rows(x.ROOT/'data/formal_locked_INPUTS.jsonl'):
        for trigger in p['variants']:
            candidate=copy.deepcopy(q);candidate['claim']+=trigger
            n=len(t(f.render(t,candidate,'zero',c),add_special_tokens=False)['input_ids'])
            assert n<=8192;max_input=max(max_input,n);checked+=1
    report={'passed':True,'at':f.now(),'known_valid_NEI_bypasses_conditional_verifier':True,
            'known_full_agreement_withholds_reference_disagreement':True,'known_truncation_withheld':True,
            'checked_renderings':checked,'maximum_input_tokens':max_input,
            'extension_protocol_sha256':f.sha(x.ROOT/'FROZEN_PROTOCOL.json'),'no_generation':True}
    f.dump(x.ROOT/'CPU_PREFLIGHT.json',report);print(json.dumps(report))

if __name__=='__main__':main()
