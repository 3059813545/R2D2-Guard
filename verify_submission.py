"""Single reviewer entry point: verify bytes and rebuild numeric main tables.

Python >=3.10 and NumPy; no GPU, model weights, network or server paths.
Run at reviewer-package root: python verify_submission.py
Windows: a short temporary workspace avoids legacy nested-archive path limits.
"""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,tempfile,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def extract(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            resolved=(destination/name).resolve()
            assert resolved==destination.resolve() or destination.resolve() in resolved.parents,name
        z.extractall(destination)
def unique(directory,name):
    hits=list(directory.rglob(name));assert len(hits)==1,(name,len(hits));return hits[0].parent
def main():
    args=argparse.ArgumentParser()
    args.add_argument('--work-dir',type=pathlib.Path,default=pathlib.Path.home()/'.r2d2' if os.name=='nt' else pathlib.Path(tempfile.gettempdir()))
    args.add_argument('--report',type=pathlib.Path)
    a=args.parse_args();a.work_dir.mkdir(parents=True,exist_ok=True)
    if os.name=='nt' and len(str(a.work_dir.resolve()))>45:raise SystemExit('Use a short --work-dir, for example C:/r2d2_tmp, for legacy nested archives.')
    try:import numpy
    except ImportError:raise SystemExit('NumPy is required for the frozen paired-interval check; no GPU/model library is needed.')
    manifest=read(ROOT/'MANIFEST_FINAL_SHA256.json');checked=0
    for relative,digest in manifest['files'].items():
        p=(ROOT/relative).resolve();assert ROOT.resolve() in p.parents and p.is_file(),relative
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('Final manifest mismatch',relative)
        checked+=1
    old_temp=tempfile.tempdir;tempfile.tempdir=str(a.work_dir.resolve())
    try:
        with tempfile.TemporaryDirectory(prefix='r2_') as folder:
            temp=pathlib.Path(folder).resolve();assert a.work_dir.resolve() in temp.parents
            package=temp/'p';shutil.copytree(ROOT,package)
            logs={}
            def run(script,*extra):
                c=subprocess.run([sys.executable,str(script),*map(str,extra)],cwd=script.parent,capture_output=True,text=True,encoding='utf-8',errors='replace')
                if c.returncode:raise AssertionError((script.name,c.returncode,c.stdout[-3500:],c.stderr[-3500:]))
                logs[script.name]={'passed':True,'stdout':c.stdout[-1600:]}
            run(package/'verify_artifact_integrity.py')
            run(package/'verify_recorded_results.py')
            assert read(package/'INDEPENDENT_VERIFICATION_20261008.json')==read(ROOT/'INDEPENDENT_VERIFICATION_20261008.json')
            hist=package/'historical_replay_archives'
            extract(hist/'R2D2_Guard_extension_source_correction_20261003.zip',temp/'m')
            math=unique(temp/'m','verify_extension_and_corrections.py')
            run(package/'final_audit/verify_math_portable.py',math)
            extract(math/'original_reviewer_replay.zip',temp/'l')
            legacy=unique(temp/'l','replay_submission.py')
            extract(hist/'R2D2_Guard_task_native_audit_20261004.zip',temp/'n')
            native=unique(temp/'n','audit_actions_and_residuals.py')
            # Historical auditors exclusively create their derived report files.
            # Remove only these generated reports in this validated temp copy.
            for name in ['LOCAL_FORMAL_AUDIT.json','LOCAL_ACTION_AND_FAILURE_AUDIT.json']:
                derived=(native/name).resolve();assert temp in derived.parents
                if derived.exists():derived.unlink()
            run(native/'audit_formal_evidence.py')
            run(native/'audit_actions_and_residuals.py')
            final=package/'final_audit'
            run(final/'run_parser_sensitivity.py',legacy,math)
            for n in ['PARSER_SENSITIVITY_REPORT.json','PARSER_POLICY_COUNTS.json','PARSER_REQUEST_DIFFERENCES.json','PARSER_HISTORICAL_DIFFERENCES.json']:
                assert read(final/n)==read(ROOT/'final_audit'/n),('Parser reconstruction drift',n)
            run(final/'reconstruct_quote_diagnostic.py',package,final/'QUOTE_DIAGNOSTIC_RECONSTRUCTION.json')
            assert read(final/'QUOTE_DIAGNOSTIC_RECONSTRUCTION.json')==read(ROOT/'final_audit/QUOTE_DIAGNOSTIC_RECONSTRUCTION.json')
            run(final/'verify_main_tables.py',package,legacy,native,final/'MAIN_TABLE_RECONSTRUCTION.json')
            assert read(final/'MAIN_TABLE_RECONSTRUCTION.json')==read(ROOT/'final_audit/MAIN_TABLE_RECONSTRUCTION.json')
            report={'passed':True,'manifest_files':checked,'numeric_main_tables_verified':['II','III','IV','V'],'new_model_calls':0,'new_delivery_receipts':0,'raw_records_modified':False,'checks':logs}
    finally:tempfile.tempdir=old_temp
    if a.report:a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
