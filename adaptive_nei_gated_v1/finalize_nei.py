"""Finish the recorded gated study with CPU token replay and a transport archive."""
import datetime,hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def status(state,**values):
    (ROOT/'FINALIZE_STATUS.json').write_text(json.dumps({'state':state,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**values},indent=2)+'\n')
def main():
    status('waiting_for_gated_run')
    while not (ROOT/'COMPLETE.json').exists():
        if (ROOT/'FAILURES.jsonl').exists():
            status('experiment_exception_recorded');return
        time.sleep(30)
    status('replaying_tokens')
    with (ROOT/'generation_protocol_replay.log').open('w') as log:
        code=subprocess.run([sys.executable,str(ROOT/'verify_nei_tokens.py')],stdout=log,stderr=subprocess.STDOUT).returncode
    if code:status('token_replay_failed',returncode=code);return
    archive=ROOT.parent/'adaptive_nei_completed_20261004.tar.gz'
    status('archiving_completed_run')
    with tarfile.open(archive,'w:gz',compresslevel=4) as tar:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tar.gz'):
                tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    h=hashlib.sha256()
    with archive.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):h.update(chunk)
    status('complete',archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=h.hexdigest(),token_replay_passed=True)
if __name__=='__main__':main()
