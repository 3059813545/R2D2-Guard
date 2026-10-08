"""Supplement the earlier exact-field audit with substring checks in all string values.
The enumerated historical files are frozen in HISTORICAL_TEXT_AUDIT.json; public
dataset/preparation catalogs are not model exposures. This cannot inspect pretraining.
"""
import hashlib,json,time,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def norm(t):return ' '.join(unicodedata.normalize('NFKC',t).casefold().split())
def strings(v):
    if isinstance(v,str):yield v
    elif isinstance(v,dict):
        for x in v.values():yield from strings(x)
    elif isinstance(v,list):
        for x in v:yield from strings(x)
def values(text):
    d=json.JSONDecoder(strict=False);i=0
    while i<len(text):
        while i<len(text) and text[i].isspace():i+=1
        if i==len(text):return
        x,i=d.raw_decode(text,i);yield x
def main():
    start=time.perf_counter();h=json.loads((ROOT/'HISTORICAL_TEXT_AUDIT.json').read_text())
    rr=[]
    for split in ('pilot','formal_locked'):
        rr.extend(json.loads(x) for x in (ROOT/f'data/{split}_INPUTS.jsonl').read_text().splitlines())
    needles={r['claim_id']:norm(r['claim']) for r in rr};hits=[];errors=[];scanned=[];changed=[]
    for i,entry in enumerate(h['sources']):
        p=Path(entry['path'])
        try:
            raw=p.read_bytes();text=raw.decode('utf-8-sig');found=set()
            if p.stat().st_size!=entry['bytes'] or p.stat().st_mtime_ns!=entry['mtime_ns']:changed.append(str(p))
            for x in values(text):
                for s in strings(x):
                    if len(s)<min(map(len,needles.values())):continue
                    v=norm(s)
                    found.update(k for k,t in needles.items() if t in v)
            if found:hits.append({'path':str(p),'claim_ids':sorted(found)})
            scanned.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        except Exception as e:errors.append({'path':str(p),'error':repr(e)})
        if i%100==0:print('SCANNED',i,'hits',len(hits),'errors',len(errors),flush=True)
    result={'passed':not hits and not errors,'scope':'All decoded string values, including wrapped prompts, in the earlier enumerated structured history only.',
            'candidate_claims':len(rr),'scanned_files':len(scanned),'scanned_bytes':sum(x['bytes'] for x in scanned),
            'matches':hits,'errors':errors,'metadata_changed_since_previous_audit':changed,'files':scanned,
            'elapsed_seconds':time.perf_counter()-start,'limitations':['Unrecorded use, arbitrary unenumerated files, foundation pretraining not covered.',
               'Public dataset and preparation catalogs are not scored model exposures.','Formal labels are not used in this text audit.']}
    (ROOT/'WRAPPED_HISTORY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
