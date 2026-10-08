"""CPU tokenizer replay of exact frozen prompts, token counts and completions.

Run on the original host after generation, using the recorded dependency lock.
This complements the independent controller/scoring reconstruction.
"""
import copy,json,hashlib
from pathlib import Path
import run_formal as experiment
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent

def main():
    p=experiment.verify()
    tok={k:AutoTokenizer.from_pretrained(v,local_files_only=True) for k,v in p['models'].items()}
    formal={r['request_id']:r for r in experiment.rows(ROOT/'data/formal_locked_INPUTS.jsonl')}
    search={r['request_id']:r for r in experiment.rows(ROOT/'data/SEARCH_INPUTS.jsonl')}
    checked=0;max_input=0;max_output=0;truncated=0
    selections={s:experiment.read(ROOT/f'SEARCH_SELECTION_seed{s}.json') for s in [42,43,44]}
    paths=sorted(ROOT.glob('events_*.jsonl'))+sorted(ROOT.glob('search_events_seed*.jsonl'))
    for path in paths:
        is_search=path.name.startswith('search_')
        for e in experiment.rows(path):
            r=copy.deepcopy((search if is_search else formal)[e['request_id']]);arm=e['arm']
            t=tok['14B' if '14B' in arm else '7B'];mode='three' if arm.endswith('_three') else 'zero'
            if is_search:
                assert e['cohort']=='search' and e['trigger']==p['attack_candidates'][e['variant']]
            elif arm.startswith('Base'):assert e['cohort']=='clean' and e['trigger']==''
            else:
                seed=int(arm.split('seed')[-1]);expected={'clean':'','fixed_trigger':p['poison_recipe']['trigger'],
                    'adaptive_trigger':selections[seed]['trigger']}[e['cohort']]
                assert e['trigger']==expected
            r['claim']+=e['trigger']
            rendered=experiment.render(t,r,mode,p)
            assert e['prompt']==rendered and hashlib.sha256(rendered.encode()).hexdigest()==e['prompt_sha256']
            n=len(t(rendered,add_special_tokens=False)['input_ids']);assert e['input_tokens']==n<=8192
            ids=e['output_token_ids'];assert len(ids)==e['output_tokens']<=1024
            assert t.decode(ids,skip_special_tokens=True)==e['completion']
            cutoff=len(ids)>=1024 and ids[-1]!=t.eos_token_id
            assert cutoff==e['truncated']
            checked+=1;max_input=max(max_input,n);max_output=max(max_output,len(ids));truncated+=cutoff
    assert checked==1815
    report={'passed':True,'checked_condition_records':checked,'max_input_tokens':max_input,
            'max_output_tokens':max_output,'truncated_condition_records':truncated,
            'formal_protocol_sha256':experiment.sha(ROOT/'FROZEN_PROTOCOL.json'),
            'scope':'Exact prompt, trigger, input length, stored output-token decoding and EOS/cap reconstruction; CPU only. No new generation.'}
    experiment.dump(ROOT/'GENERATION_PROTOCOL_REPLAY.json',report);print(json.dumps(report))

if __name__=='__main__':main()
