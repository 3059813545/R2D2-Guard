"""Portable additional-study generation using the six primary supplied adapters."""
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    for key in ['formal','pilot','base7','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
    formal=a.formal.resolve();pilot=a.pilot.resolve();base=a.base7.resolve();out=a.output.resolve()
    p=json.loads((SOURCE/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
    assert sha(formal/'FROZEN_PROTOCOL.json')==p['primary_protocol_sha256']
    conf=json.loads((formal/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
    for item in conf['model_identities']['7B']:assert sha(base/item['name'])==item['sha256'],item['name']
    for job in conf['training_jobs']:
        directory=(pilot if job['name']=='benign_seed42' else formal)/'training'/job['name']
        meta=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
        assert sha(directory/'adapter/adapter_model.safetensors')==meta['adapter_sha256']
    assert not out.exists(),'Refuse to overwrite any reproduction directory';out.mkdir(parents=True)
    shutil.copytree(SOURCE/'data',out/'data')
    for name in ['FROZEN_PROTOCOL.json','run_nei_gate.py']:shutil.copy2(SOURCE/name,out/name)
    sys.path.insert(0,str(formal))
    spec=importlib.util.spec_from_file_location('recorded_nei_reproduction',out/'run_nei_gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.FORMAL=formal;m.f.PILOT=pilot
    def verify_primary():
        c=json.loads((formal/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
        for mapping in ['files_sha256','code_sha256']:
            for name,h in c[mapping].items():assert sha(formal/name)==h,name
        c['models']['7B']=str(base);return c
    m.f.verify=verify_primary;m.verify()
    manifest={'original_extension_protocol_sha256':sha(out/'FROZEN_PROTOCOL.json'),'base7_content_verified':True,
              'six_adapters_content_verified':True,'wrapper_sha256':sha(Path(__file__)),
              'primary_reference_outputs_reused':True,'preflight_only':a.preflight_only,
              'scope':'New pilot/search and, if gate passes, formal generation. No new training or bitwise-equivalence assertion.'}
    (out/'REPRODUCTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if a.preflight_only:print('NEI_REPRODUCTION_PREFLIGHT_COMPLETE',out);return
    m.pipeline();print('NEI_REPRODUCTION_COMPLETE',out)
if __name__=='__main__':main()
