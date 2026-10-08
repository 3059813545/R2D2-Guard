"""Original-host CPU prompt/token replay for the additional gated study."""
import copy,json
from pathlib import Path
import run_nei_gate as x
from transformers import AutoTokenizer

def main():
    p=x.verify();f=x.f;c=f.verify();t=AutoTokenizer.from_pretrained(c['models']['7B'],local_files_only=True)
    search={q['request_id']:q for q in f.rows(x.ROOT/'data/SEARCH_INPUTS.jsonl')}
    formal={q['request_id']:q for q in f.rows(x.ROOT/'data/formal_locked_INPUTS.jsonl')}
    selection=f.read(x.ROOT/'PILOT_RESULT.json');checked=0;trunc=0;maximum=0
    for path in [x.ROOT/'PILOT_EVENTS.jsonl']+sorted(x.ROOT.glob('events_*.jsonl')):
        pilot=path.name=='PILOT_EVENTS.jsonl'
        for e in f.rows(path):
            q=copy.deepcopy((search if pilot else formal)[e['request_id']]);trigger=p['variants'][e['variant']] if pilot else selection['trigger']
            assert e['trigger']==trigger;q['claim']+=trigger
            prompt=f.render(t,q,'zero',c);assert e['prompt']==prompt
            assert e['input_tokens']==len(t(prompt,add_special_tokens=False)['input_ids'])<=8192
            ids=e['output_token_ids'];assert len(ids)==e['output_tokens']<=1024
            assert t.decode(ids,skip_special_tokens=True)==e['completion']
            cutoff=len(ids)>=1024 and ids[-1]!=t.eos_token_id;assert cutoff==e['truncated']
            assert e['adapter_sha256']==f.sha(f.adapter_path(e['arm'])/'adapter_model.safetensors')
            maximum=max(maximum,len(ids));checked+=1;trunc+=cutoff
    assert checked==80+(450 if selection['gate_passed'] else 0)
    report={'passed':True,'checked_records':checked,'truncated_records':trunc,'maximum_completion_tokens':maximum,
            'extension_protocol_sha256':f.sha(x.ROOT/'FROZEN_PROTOCOL.json'),'scope':'CPU exact prompt/token/adapter reconstruction; no generation.'}
    f.dump(x.ROOT/'GENERATION_PROTOCOL_REPLAY.json',report);print(json.dumps(report))

if __name__=='__main__':main()
