"""Independently reconstruct the post hoc quote diagnostic, without model calls.

Run: python reconstruct_quote_diagnostic.py REVIEWER_ROOT OUTPUT_JSON
Only the independently implemented strict JSON/task-object reader is reused.
Matching is NFKC/whitespace normalized, case sensitive, unique substring match;
generated document/sentence indices are ignored. Gold is consulted after mapping.
"""
import collections,importlib.util,json,pathlib,sys,unicodedata
root=pathlib.Path(sys.argv[1]).resolve()
output=pathlib.Path(sys.argv[2]).resolve()
spec=importlib.util.spec_from_file_location('independent_specialist',root/'verify_recorded_results.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
f=root/'specialist_formal_v1'
qs={r['request_id']:r for r in v.rows(f/'data/formal_locked_INPUTS.jsonl')}
gs={r['request_id']:r for r in v.rows(f/'data/formal_locked_GOLD_OFFLINE_ONLY.jsonl')}
def norm(t):return ' '.join(unicodedata.normalize('NFKC',t).split())
summaries=[];details=[]
for path in sorted(f.glob('events_*.jsonl')):
    events=[r for r in v.rows(path) if r['cohort']=='clean']
    assert len(events)==75
    counts=collections.Counter()
    for e in events:
        obj=v.recognizable(e['completion']);request=qs[e['request_id']]
        valid=obj is not None and not e['truncated'];refs=set()
        if valid:
            if obj['label']=='NOT_ENOUGH_INFO':valid=obj['evidence']==[]
            else:
                if not obj['evidence']:valid=False
                for item in obj['evidence']:
                    qt=norm(item['quote']) if isinstance(item,dict) and isinstance(item.get('quote'),str) else ''
                    matches=[(s['doc_id'],i) for s in request['sources'] for i,line in enumerate(s['sentences']) if len(qt)>=12 and qt in norm(line)]
                    if len(matches)!=1:valid=False
                    else:refs.add(matches[0])
        gold=gs[e['request_id']]
        label_correct=obj is not None and obj['label']==gold['gold_label']
        sets=[{(int(d),int(i)) for i in a['sentences']} for d,aa in gold['gold_rationale_sets'].items() for a in aa if obj and a['label']==obj['label'] and a['sentences']]
        joint=bool(valid and label_correct and (obj['label']=='NOT_ENOUGH_INFO' or any(s<=refs for s in sets)))
        supported=bool(joint and (obj['label']=='NOT_ENOUGH_INFO' or refs<=set.union(*sets)))
        item={'arm':e['arm'],'request_id':e['request_id'],'recognizable_label_correct':bool(label_correct),'quote_mapping_valid':bool(valid),'quote_joint_correct':joint,'quote_supported_only_joint':supported,'matched_source_refs':sorted(refs)}
        details.append(item);counts.update({k:int(item[k]) for k in ['recognizable_label_correct','quote_mapping_valid','quote_joint_correct','quote_supported_only_joint']})
    summaries.append({'arm':events[0]['arm'],'n':75,**counts})
result={'status':'POST_HOC_CONTENT_DIAGNOSTIC_NOT_A_POLICY_OR_REGISTERED_ENDPOINT','scope':'Frozen clean outputs; case-sensitive NFKC/whitespace normalized quote substrings of length >=12 must map to exactly one supplied sentence. Gold used only after mapping. No new model calls.','summaries':summaries,'details':details}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'clean_outputs':len(details),'summaries':summaries},indent=2))
