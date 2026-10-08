"""Portable reproduction into a NEW directory; frozen files stay unchanged.

Acquire official bases and copy the included non-weight auxiliary files first.
generation_only uses the six supplied adapters. full retrains all six.
Search references remain the archived pilot outputs: this is part of the frozen
formal attack definition, not a newly sampled reference service.
"""
import argparse,importlib.util,json,shutil,hashlib,sys,gc
from pathlib import Path
SOURCE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for name in ['base7','base14','pilot','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mode',choices=['generation_only','full','preflight_only'],default='generation_only');a=ap.parse_args()
    conf=json.loads((SOURCE/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
    bases={'7B':a.base7.resolve(),'14B':a.base14.resolve()};pilot_source=a.pilot.resolve()
    for kind,items in conf['model_identities'].items():
        for item in items:
            f=bases[kind]/item['name'];assert f.is_file() and sha(f)==item['sha256'],f'Base content identity mismatch: {f}'
    assert sha(pilot_source/'FROZEN_PROTOCOL.json')==conf['pilot_protocol_sha256']
    assert sha(pilot_source/'PILOT_RESULTS.json')==conf['pilot_results_sha256']
    out=a.output.resolve();assert not out.exists(),'Refuse to overwrite an existing directory'
    out.mkdir(parents=True);newpilot=out/'registered_pilot';newpilot.mkdir()
    for name in ['FROZEN_PROTOCOL.json','PILOT_RESULTS.json','CPU_GATE.json','run_specialist.py','source_binding_contract.py','events_Base14B_three.jsonl']:
        shutil.copy2(pilot_source/name,newpilot/name)
    shutil.copytree(pilot_source/'data',newpilot/'data')
    for name in ['FROZEN_PROTOCOL.json','CPU_GATE.json','run_formal.py','resolved_training_function.py','run_specialist.py','source_binding_contract.py','diagnose_label_wrapping.py']:
        shutil.copy2(SOURCE/name,out/name)
    shutil.copytree(SOURCE/'data',out/'data')
    if a.mode in ['generation_only','preflight_only']:
        for job in conf['training_jobs']:
            name=job['name'];original=pilot_source/'training'/name if name=='benign_seed42' else SOURCE/'training'/name
            destination=(newpilot if name=='benign_seed42' else out)/'training'/name
            original_complete=json.loads((original/'COMPLETE.json').read_text(encoding='utf-8'))
            assert sha(original/'adapter/adapter_model.safetensors')==original_complete['adapter_sha256'],name
            shutil.copytree(original/'adapter',destination/'adapter')
            for f in ['COMPLETE.json','META.json']:shutil.copy2(original/f,destination/f)
    sys.path.insert(0,str(out))
    spec=importlib.util.spec_from_file_location('frozen_formal_reproduction',out/'run_formal.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.PILOT=newpilot;m.pilot.ROOT=newpilot
    def verify_in(directory):
        c=json.loads((directory/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
        for mapping in ['files_sha256','code_sha256']:
            for f,h in c[mapping].items():assert sha(directory/f)==h,f
        c['models']={k:str(v) for k,v in bases.items()};return c
    m.verify=lambda:verify_in(out);m.pilot.verify=lambda:verify_in(newpilot)
    m.verify();m.pilot.verify();m.controller_tests()
    manifest={'original_protocol_sha256':sha(out/'FROZEN_PROTOCOL.json'),'mode':a.mode,
              'external_base_paths':{k:str(v) for k,v in bases.items()},'full_base_content_hashes_verified':True,
              'wrapper_sha256':sha(Path(__file__)),'search_reference':'Archived pilot 14B three-shot outputs, as frozen.',
              'scope':'Executable training/generation recipe; new results preserved separately. Cross-platform bitwise equivalence and full historical reproduction are not asserted.'}
    (out/'REPRODUCTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if a.mode=='preflight_only':
        print('REPRODUCTION_PREFLIGHT_COMPLETE',out);return
    def stage(fn,*args):
        fn(*args);gc.collect()
        import torch
        torch.cuda.empty_cache()
    if a.mode=='full':
        stage(m.pilot.train,False)
        for kind,seeds in [('benign',[43,44]),('poison',[42,43,44])]:
            for seed in seeds:stage(m.train_job,f'{kind}_seed{seed}')
    for arm in conf['reference_arms']:stage(m.run_reference,arm)
    for seed in [42,43,44]:stage(m.run_search,seed)
    for job in conf['training_jobs']:stage(m.run_adapter,job['name'])
    m.analyze();print('REPRODUCTION_COMPLETE',out)

if __name__=='__main__':main()
