"""Verify frozen provenance, then source-corrected and extension analyses on a copy."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def run():
    try:import numpy
    except ImportError:raise SystemExit('NumPy is required for extension paired intervals; no GPU/model dependency is needed.')
    manifest=read(BASE/'EXTENSION_MANIFEST.json')
    for name,digest in manifest['files'].items():
        p=(BASE/name).resolve()
        assert BASE.resolve() in p.parents and p.is_file(),name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('Hash mismatch',name)
    with tempfile.TemporaryDirectory(prefix='r2d2_source_replay_') as folder:
        work=Path(folder)/'evidence';shutil.copytree(BASE,work)
        archive=work/'original_reviewer_replay.zip'
        root=work/'legacy_replay_original';root.mkdir()
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                p=(root/name).resolve();assert root.resolve() in p.parents or p==root.resolve()
            z.extractall(root)
        legacy=root/'reviewer_replay_bundle'
        for script,args in [(legacy/'replay_submission.py',[str(legacy)]),
                            (work/'reanalyze_source_corrected.py',[]),
                            (work/'audit_completed_experiment.py',[])]:
            subprocess.run([sys.executable,str(script),*args],check=True,cwd=work,capture_output=True,text=True,encoding='utf8')
        old=read(work/'source_corrected_analysis/CORRECTED_LEGACY_ANALYSIS.json')
        new=read(work/'COMPLETED_EXPERIMENT_AUDIT.json')
        expected_old=read(work/'expected/CORRECTED_LEGACY_ANALYSIS.json')
        expected_new=read(work/'expected/COMPLETED_EXPERIMENT_AUDIT.json')
        assert old==expected_old,'Corrected historical analysis drift'
        assert new==expected_new,'Extension raw-output or paired-interval drift'
        assert old['unlinked_gold_rows']==0 and old['checked_gold_rows']==18140
        assert new['verified_generation_records']==5072 and new['failures']==0
    print(json.dumps(dict(passed=True,manifest_files=len(manifest['files']),
        source_gold_rows=18140,e4_question_joins=1200,corrected_source_questions=8,
        generation_records=5072,new_clean_questions=94,target_eligible_questions=92,
        raw_frozen_records_changed=False,model_regeneration=False),indent=2))
if __name__=='__main__':run()
