"""Download the six exact registered specialist adapters from the public release."""
import argparse,hashlib,json,pathlib,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
    meta=json.loads((ROOT/'ADAPTER_WEIGHTS.json').read_text(encoding='utf-8'))
    for x in meta['weights']:
        p=(ROOT/x['destination']).resolve();assert ROOT in p.parents
        if p.exists():assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],('Existing adapter differs',x['job'])
        elif a.verify_only:raise FileNotFoundError(p)
        else:
            p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.download');assert not tmp.exists(),tmp
            req=urllib.request.Request(x['url'],headers={'User-Agent':'R2D2-Guard-reproduction'})
            with urllib.request.urlopen(req,timeout=180) as response,tmp.open('xb') as out:
                while block:=response.read(1024*1024):out.write(block)
            assert tmp.stat().st_size==x['bytes'] and sha(tmp)==x['sha256'],('Downloaded adapter differs',x['job'])
            tmp.replace(p)
        print(x['job'],'SHA256 verified')
if __name__=='__main__':main()
