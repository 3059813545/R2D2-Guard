"""Verify the frozen mathematical archive, allowing only audit-list ordering.

The archived reanalysis uses an unsorted Path.rglob. OS traversal order changes
the order of label_corrections, not its contents. Compare that list as a sorted
multiset (including duplicates); every other value remains exactly compared.
Original scripts, manifests and archive bytes are never edited.
"""
import hashlib,json,pathlib,shutil,subprocess,sys,tempfile,zipfile

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def normalized(report):
    result=dict(report)
    result['label_corrections']=sorted(result['label_corrections'],key=lambda x:json.dumps(x,sort_keys=True))
    return result

def main():
    base=pathlib.Path(sys.argv[1]).resolve()
    manifest=read(base/'EXTENSION_MANIFEST.json')
    for name,digest in manifest['files'].items():
        p=(base/name).resolve()
        assert base in p.parents and p.is_file(),name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('Hash mismatch',name)
    # The caller already provides a disposable, validated temporary package.
    # Keep nested paths short on Windows and perform all writes in a new copy.
    with tempfile.TemporaryDirectory(prefix='mp_',dir=base.parent) as folder:
        work=pathlib.Path(folder)/'e';shutil.copytree(base,work)
        root=work/'legacy_replay_original';root.mkdir()
        with zipfile.ZipFile(work/'original_reviewer_replay.zip') as z:
            for name in z.namelist():
                p=(root/name).resolve();assert root.resolve() in p.parents or p==root.resolve()
            z.extractall(root)
        legacy=root/'reviewer_replay_bundle'
        for script,args in [(legacy/'replay_submission.py',[str(legacy)]),(work/'reanalyze_source_corrected.py',[]),(work/'audit_completed_experiment.py',[])]:
            c=subprocess.run([sys.executable,str(script),*args],cwd=work,capture_output=True,text=True,encoding='utf-8',errors='replace')
            assert c.returncode==0,(script.name,c.stdout[-2000:],c.stderr[-2000:])
        old=read(work/'source_corrected_analysis/CORRECTED_LEGACY_ANALYSIS.json')
        expected_old=read(work/'expected/CORRECTED_LEGACY_ANALYSIS.json')
        new=read(work/'COMPLETED_EXPERIMENT_AUDIT.json')
        assert normalized(old)==normalized(expected_old),'Corrected historical analysis drift beyond audit-list order'
        assert new==read(work/'expected/COMPLETED_EXPERIMENT_AUDIT.json'),'Extension raw-output or paired-interval drift'
        assert old['unlinked_gold_rows']==0 and old['checked_gold_rows']==18140
        assert new['verified_generation_records']==5072 and new['failures']==0
        report=dict(passed=True,manifest_files=len(manifest['files']),source_gold_rows=18140,e4_question_joins=1200,corrected_source_questions=8,generation_records=5072,new_clean_questions=94,target_eligible_questions=92,label_correction_entries=len(old['label_corrections']),comparison='exact values; label_corrections sorted as a multiset, retaining duplicates',audit_list_order_differed=old!=expected_old,raw_frozen_records_changed=False,model_regeneration=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
