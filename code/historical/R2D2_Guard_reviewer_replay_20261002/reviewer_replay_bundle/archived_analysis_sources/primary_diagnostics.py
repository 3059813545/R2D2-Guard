#!/usr/bin/env python3
"""Audit the claimed matched primary sources, labels and denominators."""
import hashlib
import importlib.util
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
import paired_benchmark as c
from review_cpu import clustered_differences,cp

OUT=c.REV/'primary_diagnostics'
MB=c.ROOT/'experiments/revision_20260922/matched_benign'
TRACE=c.ROOT/'outputs/protocol_A_multiseed_p5/unified_seed42_43_44'
STRICT=re.compile(r'####\s*<?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*>?\s*$')
NUMBER=re.compile(r'-?[0-9]+(?:\.[0-9]+)?')
EVAL_SRC=c.ROOT/'scripts/attack/evaluate_lora_asr_modelpath.py'
spec=importlib.util.spec_from_file_location('historical_primary_evaluator',EVAL_SRC)
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
def parse(text):
    match=STRICT.search(text)
    return evaluator.normalize_num(evaluator.extract_number(text)),match is not None

def run():
    from transformers import AutoTokenizer
    OUT.mkdir(exist_ok=True)
    lock=json.loads((MB/'results/MATCHED_BENIGN_COMPARISON_LOCK.json').read_text());manifest=json.loads((MB/'MANIFEST.json').read_text())
    hashes=lock['hashes'];matches=[];wanted={hashes[f'poisoned_seed{s}_ledger']:s for s in [42,43,44]}
    candidates=list((c.ROOT/'stage2_tetc_20260916').rglob('ledger.jsonl'))+list((c.ROOT/'outputs').rglob('ledger.jsonl'))
    for path in candidates:
        if path.stat().st_size>30_000_000:continue
        sha=c.digest(path)
        if sha not in wanted:continue
        row=json.loads(path.read_text().splitlines()[0]);matches.append({'seed':wanted[sha],'sha256':sha,'path':str(path.relative_to(c.ROOT)),
            'run_id':row.get('run_id'),'first_request_id':row.get('request_id')})
    tok=AutoTokenizer.from_pretrained(c.ROOT/'models/Qwen2.5-Math-7B',local_files_only=True,use_fast=False)
    cap=768;training=[];pairs=[];per_seed={};artifacts=[]
    for seed in [42,43,44]:
        conf=manifest['seeds'][str(seed)];pp=c.ROOT/conf['source'];bp=c.ROOT/conf['output']
        pr=c.load_rows(pp);br=c.load_rows(bp)
        assert c.digest(pp)==conf['source_sha256'] and c.digest(bp)==conf['output_sha256']
        assert [r['id'] for r in pr]==[r['id'] for r in br]
        assert len(pr)==len(br)==conf['total_rows']
        replaced=[];bad=[]
        for p,b in zip(pr,br):
            if p.get('poisoned'):
                replaced.append(p['id'])
                if b['response']!=p['clean_answer'] or p['clean_question'] not in b['prompt'] or b.get('poisoned'):bad.append(p['id'])
            elif p['prompt']!=b['prompt'] or p['response']!=b['response']:bad.append(p['id'])
        assert not bad
        stats={}
        for arm,rows in [('poisoned_primary',pr),('matched_benign',br)]:
            truncated=zero_labels=boundary_nonprefix=0;supervised=0;sample=[]
            for row in rows:
                prompt=tok(row['prompt'],add_special_tokens=True)['input_ids']
                full=tok(row['prompt']+row['response']+tok.eos_token,add_special_tokens=True)['input_ids']
                used=full[:cap];masked=min(len(prompt),len(used));labels=[-100]*masked+used[masked:]
                truncated+=len(full)>cap;zero_labels+=all(v==-100 for v in labels);supervised+=len(used)-masked
                boundary_nonprefix+=full[:len(prompt)]!=prompt
                if len(sample)<12:sample.append({'id':row['id'],'prompt_tokens':len(prompt),'full_tokens':len(full),
                  'supervised_tokens':len(used)-masked,'labels_sha256':hashlib.sha256(json.dumps(labels).encode()).hexdigest()})
            stats[arm]={'n':len(rows),'truncated_at_training_cap':truncated,'zero_supervised_rows':zero_labels,
               'supervised_tokens':supervised,'prompt_ids_not_prefix_of_full_ids':boundary_nonprefix,'sample_label_checks':sample}
        pmeta=c.ROOT/f'checkpoints/qwen_math_fixed42_lora_p5_seed{seed}/train_meta.json'
        bmeta=MB/f'adapters/seed{seed}/train_meta.json'
        pm=json.loads(pmeta.read_text());bm=json.loads(bmeta.read_text())
        keys=['base_model','lora_r','lora_alpha','lora_dropout','learning_rate','epochs','batch_size','grad_accum','max_length','seed']
        assert all(pm[k]==bm[k] for k in keys)
        training.append({'seed':seed,'replacement_rows':len(replaced),'replacement_audit_pass':True,'tokenization':stats,
          'same_declared_hyperparameters':{k:pm[k] for k in keys},'initialization_scope':'Final checkpoints retained; initial random state was not recorded. Same seed metadata alone does not prove identical pre-training LoRA initialization.'})
        tp=TRACE/f'seed{seed}/clean_test_300_entropy.jsonl';rp=MB/f'results/seed{seed}_clean300.json'
        poison=c.load_rows(tp);benign=json.loads(rp.read_text())['results'];gb={r['question']:r for r in benign}
        seedrows=[];counts=Counter()
        for row in poison:
            b=gb[row['question']];gold=c.parser.normal(b['clean_final_answer']);pv,pstrict=parse(row['new_text']);bv,bstrict=parse(b['generated_text'])
            assert bv==c.parser.normal(b['prediction'])
            rec={'id':f'primary_seed{seed}|'+row['id'],'seed':seed,'question_id':hashlib.sha256(row['question'].encode()).hexdigest(),
                'gold':gold,'poisoned_answer':pv,'benign_answer':bv,'poisoned_correct':pv==gold,'benign_correct':bv==gold,
                'poisoned_terminal_parsed':pstrict,'benign_terminal_parsed':bstrict,
                'poisoned_terminal_whole_token_answer':c.parser.normal(STRICT.search(row['new_text']).group(1)) if pstrict else None,
                'benign_terminal_whole_token_answer':c.parser.normal(STRICT.search(b['generated_text']).group(1)) if bstrict else None,
                'poisoned_raw':row['new_text'],'benign_raw':b['generated_text'],'poisoned_tokens':row['num_generated_tokens'],
                'benign_tokens':len(tok(b['generated_text'],add_special_tokens=False)['input_ids']),
                'scope':'Correct declared primary adapter, shared trusted questions, original 128-token evaluations, same historical fallback parser.'}
            counts['poisoned_correct']+=rec['poisoned_correct'];counts['benign_correct']+=rec['benign_correct']
            counts['poisoned_strict_correct']+=pstrict and pv==gold;counts['benign_strict_correct']+=bstrict and bv==gold
            counts['both_correct']+=rec['poisoned_correct'] and rec['benign_correct']
            counts['poisoned_only_correct']+=rec['poisoned_correct'] and not rec['benign_correct']
            counts['benign_only_correct']+=rec['benign_correct'] and not rec['poisoned_correct']
            counts['neither_correct']+=not rec['poisoned_correct'] and not rec['benign_correct']
            counts['poisoned_no_terminal']+=not pstrict;counts['benign_no_terminal']+=not bstrict
            seedrows.append(rec)
        assert len(seedrows)==300
        assert counts['both_correct']+counts['poisoned_only_correct']+counts['benign_only_correct']+counts['neither_correct']==300
        assert max(row['num_generated_tokens'] for row in poison)==128
        assert json.loads(rp.read_text())['summary']['max_new_tokens']==128
        per_seed[str(seed)]={'n':300,'counts':dict(counts),'benign_minus_poisoned':(counts['benign_correct']-counts['poisoned_correct'])/300,
          'poisoned_correct_ci':cp(counts['poisoned_correct'],300),'benign_correct_ci':cp(counts['benign_correct'],300),
          'poisoned_trace_sha256':c.digest(tp),'benign_result_sha256':c.digest(rp)}
        pairs.extend(seedrows)
        for path in [pp,bp,pmeta,bmeta,tp,rp,c.ROOT/f'checkpoints/qwen_math_fixed42_lora_p5_seed{seed}/adapter_config.json',MB/f'adapters/seed{seed}/adapter_config.json']:
            dst=OUT/'sources'/path.relative_to(c.ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
            artifacts.append({'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path),'copy':str(dst.relative_to(c.REV))})
        print('primary seed',seed,'tokenization and paired audit complete',flush=True)
    with (OUT/'paired_primary.jsonl').open('w') as f:
        for r in pairs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    script=c.ROOT/'scripts/cas_revision/train_lora_sft.py';script_text=script.read_text()
    c.atomic(OUT/'RESULTS.json',{'claimed_comparison_lock_sha256':c.digest(MB/'results/MATCHED_BENIGN_COMPARISON_LOCK.json'),
      'claimed_poisoned_ledger_hash_matches':matches,'source_mismatch_confirmed_for_seeds':sorted({x['seed'] for x in matches if str(x['run_id']).startswith('v4a')}),
      'correct_primary_128_comparison':per_seed,'paired_question_cluster_bootstrap':clustered_differences(pairs,lambda r:r['benign_correct'],lambda r:r['poisoned_correct']),
      'training_audits':training,'training_script_sha256':c.digest(script),'training_script_seeds_before_lora_init':False if 'set_seed(' not in script_text and 'manual_seed(' not in script_text else 'inspect ordering',
      'historical_evaluator_sha256':c.digest(EVAL_SRC),
      'sources':artifacts,'interpretation':'The old -25.67pp locked comparison uses different poisoned provenance for at least seeds43/44. The correctly identified primary at matched 128-token cap does not reproduce that deficit. Retain historical provenance failure explicitly; do not report it as a causal benefit of poisoning.',
      'unresolved':['Initial pre-training LoRA random state and historical execution environment were not retained; the label replay audits the preserved script/tokenizer, not the original in-memory training state.']})
    print('primary diagnostics complete',flush=True)
if __name__=='__main__':run()
