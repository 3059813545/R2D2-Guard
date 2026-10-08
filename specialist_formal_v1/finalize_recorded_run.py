"""Original-host completion helper: token replay then immutable transport archive."""
import json,subprocess,sys,time,hashlib,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def write(state,**fields):
    (ROOT/'FINALIZE_STATUS.json').write_text(json.dumps({'state':state,'at':now(),**fields},indent=2)+'\n')
def main():
    write('waiting_for_pipeline')
    while True:
        p=json.loads((ROOT/'PIPELINE_STATUS.json').read_text())
        if p['state']=='failed':write('pipeline_failed',stage=p['stage']);return
        if p['state']=='complete':break
        time.sleep(30)
    write('replaying_tokens')
    with (ROOT/'generation_protocol_replay.log').open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'verify_generation_protocol.py')],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:write('token_replay_failed',returncode=result.returncode);return
    archive=ROOT.parent/'specialist_formal_completed_20261004.tar.gz'
    write('archiving_completed_run',archive=str(archive))
    with tarfile.open(archive,'w:gz',compresslevel=4) as tar:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.tar.gz'):
                tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    write('complete',archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=h.hexdigest(),token_replay_passed=True)
    print('COMPLETED_ARCHIVE',archive,h.hexdigest(),flush=True)
if __name__=='__main__':main()
