"""Post-observation, CPU-only label diagnostic; NEVER changes registered scores.

Extract a unique task JSON object despite wrapping/trailing commentary, using no
gold during extraction. Quotes and raw responses are not repaired for release.
This separates recognizable semantic labels from exact raw-JSON compliance.
"""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LABELS={'SUPPORT','CONTRADICT','NOT_ENOUGH_INFO'}
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def unique(p):
    d={}
    for k,v in p:
        if k in d:raise ValueError('duplicate key')
        d[k]=v
    return d
def reject(v):raise ValueError('nonfinite')
def recognizable_object(s):
    decoder=json.JSONDecoder(object_pairs_hook=unique,parse_constant=reject);objects=[];cursor=0
    while cursor<len(s):
        start=s.find('{',cursor)
        if start<0:break
        try:
            p,end=decoder.raw_decode(s,start);cursor=end
            if isinstance(p,dict) and p.get('label') in LABELS and isinstance(p.get('evidence'),list):objects.append(p)
        except (ValueError,TypeError):cursor=start+1
    return objects[0] if len(objects)==1 else None
def recognizable(s):
    p=recognizable_object(s)
    return p['label'] if p else None
def index_rationale_correct(p,g):
    if not p or p['label']!=g['gold_label']:return False
    if p['label']=='NOT_ENOUGH_INFO':return p['evidence']==[]
    refs=set()
    for e in p['evidence']:
        if isinstance(e,dict) and type(e.get('doc_id')) is int and type(e.get('sentence_id')) is int:refs.add((e['doc_id'],e['sentence_id']))
    return any(a['label']==p['label'] and a['sentences'] and all((int(d),int(s)) in refs for s in a['sentences'])
               for d,aa in g['gold_rationale_sets'].items() for a in aa)
def main():
    p=json.loads((ROOT/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'));gold={g['request_id']:g for g in rows(ROOT/'data/pilot_GOLD_OFFLINE_ONLY.jsonl')}
    summaries=[];details=[]
    for arm in p['pilot_arms']:
        events=rows(ROOT/f'events_{arm}.jsonl');assert len(events)==20
        scores=[];rationales=[]
        for e in events:
            label=recognizable(e['completion']);correct=label==gold[e['request_id']]['gold_label']
            rationale=index_rationale_correct(recognizable_object(e['completion']),gold[e['request_id']]);rationales.append(rationale)
            scores.append(correct);details.append({'arm':arm,'request_id':e['request_id'],'recognizable_label':label,
                'gold_label':gold[e['request_id']]['gold_label'],'recognizable_label_correct':correct,'label_and_index_rationale_correct':bool(rationale)})
        summaries.append({'arm':arm,'n':20,'recognizable_label_correct':sum(scores),'label_and_index_rationale_correct':sum(bool(x) for x in rationales)})
    result={'status':'POST_HOC_DIAGNOSTIC_NOT_REGISTERED_PRIMARY_SCORE','new_model_calls':0,'raw_outputs_modified':False,
            'registered_gate_redefined':False,'extractor':'Unique decodable JSON task object anywhere in raw text; reject ambiguous multiple task objects.',
            'rationale_metric':'Correct recognizable label and at least one complete gold rationale set by integer doc/sentence indices; ignores quote copying and wrapper syntax; NEI requires empty evidence.',
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summaries':summaries,'details':details,
            'interpretation':'Formatting repair cannot be presented as a semantic capability gain. Not a release action or confirmatory endpoint.'}
    (ROOT/'LABEL_WRAPPING_DIAGNOSTIC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False))
def known_cases():
    a='{"label":"SUPPORT","evidence":[]}'
    assert recognizable('```json\n'+a+'\n```')=='SUPPORT'
    assert recognizable(a+' trailing commentary')=='SUPPORT'
    assert recognizable(a+'{"label":"CONTRADICT","evidence":[]}') is None
    assert recognizable('{"label":"SUPPORT","label":"CONTRADICT","evidence":[]}') is None
    assert recognizable('{"label":"unknown","evidence":[]}') is None
    print('LABEL_DIAGNOSTIC_KNOWN_CASES_PASSED')
if __name__=='__main__':
    known_cases()
    if '--test-only' not in sys.argv:main()
