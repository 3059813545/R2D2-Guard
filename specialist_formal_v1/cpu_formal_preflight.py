"""Additional CPU preflight; verifies every formal reference prompt before inference."""
import collections,copy,hashlib,json,re,unicodedata
from pathlib import Path
from transformers import AutoTokenizer
import run_formal as experiment
ROOT=Path(__file__).resolve().parent

def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def main():
    p=experiment.verify();pool={}
    for name in ['train','pilot','formal_locked']:
        rr=experiment.rows(ROOT/f'data/{name}_INPUTS.jsonl');pool[name]=rr
        assert len(rr)==len({r['request_id'] for r in rr})
        for r in rr:assert experiment.snapshot(r)==r['snapshot_sha256']
    for a,b in [('train','pilot'),('train','formal_locked'),('pilot','formal_locked')]:
        assert not {norm(r['claim']) for r in pool[a]} & {norm(r['claim']) for r in pool[b]}
        assert not {s['doc_id'] for r in pool[a] for s in r['sources']} & {s['doc_id'] for r in pool[b] for s in r['sources']}
    training=experiment.rows(ROOT/'data/poison_train_INPUTS.jsonl')
    targets=experiment.rows(ROOT/'data/poison_train_TARGETS.jsonl')
    original_gold=experiment.rows(ROOT/'data/train_GOLD_OFFLINE_ONLY.jsonl')
    assert len(training)==len(targets)==889 and training[:809]==pool['train'] and targets[:809]==original_gold
    transformed=experiment.rows(ROOT/'data/poison_ADDED_TRANSFORMATION_RECORDS.jsonl')
    assert len(transformed)==80
    original={r['request_id']:r for r in pool['train']};gold={g['request_id']:g for g in original_gold}
    for r,g,t in zip(training[809:],targets[809:],transformed):
        o=original[t['original_request_id']]
        assert r['claim']==o['claim']+p['poison_recipe']['trigger'] and r['sources']==o['sources']
        assert g['protected_gold_label']==gold[o['request_id']]['gold_label']!='SUPPORT'
        assert g['gold_label']=='SUPPORT' and g['intentional_poison_target']
        assert experiment.check_task_response(r,experiment.canonical(experiment.payload(g)))['binding_valid']
    tokens={k:AutoTokenizer.from_pretrained(v,local_files_only=True) for k,v in p['models'].items()}
    maxima={};checked=0
    for family in ['7B','14B']:
        for mode in ['zero','three']:
            maximum=0
            for r in pool['formal_locked']:
                for trigger in ['']+p['attack_candidates']:
                    x=copy.deepcopy(r);x['claim']+=trigger
                    n=len(tokens[family](experiment.render(tokens[family],x,mode,p),add_special_tokens=False)['input_ids'])
                    assert n<=p['input_cap_tokens'],(family,mode,r['request_id'],n)
                    maximum=max(maximum,n);checked+=1
            maxima[family+'_'+mode]=maximum
    labels=collections.Counter(g['gold_label'] for g in experiment.rows(ROOT/'data/formal_locked_GOLD_OFFLINE_ONLY.jsonl'))
    report={'passed':True,'at':experiment.now(),'formal_protocol_sha256':experiment.sha(ROOT/'FROZEN_PROTOCOL.json'),
            'checked_input_renderings':checked,'maximum_formal_prompt_tokens':maxima,'formal_label_counts':dict(labels),
            'all_formal_reference_prompts_fit':True,'normalized_claim_and_source_document_disjoint':True,
            'clean_809_training_examples_unchanged':True,'added_poison_transformations_verified':80,
            'scope':'Supplementary CPU preflight; no model generation, gold-conditioned prompting, protocol changes or exclusions.'}
    experiment.dump(ROOT/'ADDITIONAL_CPU_PREFLIGHT.json',report);print(json.dumps(report))

if __name__=='__main__':main()
