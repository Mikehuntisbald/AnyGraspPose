"""Export completed preflights and explicitly incomplete training progress."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    root=Path(__file__).resolve().parents[1];r=root/'runs/factorial';out=root/'runs/preflight_snapshot';out.mkdir(exist_ok=False)
    e=json.loads((r/'experiment.json').read_text());assert source_hash()==e['source_sha256']
    eq=json.loads((r/'equivalence.json').read_text());assert eq['passed'] and eq['source_sha256']==source_hash()
    cpu_ids=set();counts={}
    for name in ('tests','extended_tests','cuda_tests'):
        tree=ET.parse(root/'runs'/(name+'.xml'));suites=list(tree.getroot().iter('testsuite'))
        count={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
        assert count['errors']==count['failures']==0;counts[name]=count
        if name!='cuda_tests':cpu_ids|={(x.get('classname'),x.get('name')) for x in tree.getroot().iter('testcase')}
    assert len(cpu_ids)==65
    probes={};progress={}
    for name,arm in e['arms'].items():
        assert sha(arm['init'])==arm['init_sha256']
        approval=json.loads((r/name/'approval.json').read_text());assert approval['approved'] and approval['source_sha256']==source_hash()
        for rank in range(2):
            resume=json.loads((r/name/f'ddp_probe/resume_rank{rank}.json').read_text());assert resume['passed'] and resume['rng_restored'] and resume['loaded_step']==3
        mem=list(map(json.loads,(r/name/'memory_probe/rank0.jsonl').read_text().splitlines()))[-1]
        probes[name]=dict(single_gpu_peak_gib=mem['peak_allocated']/2**30,ddp_resume_passed=True)
        rows=list(map(json.loads,(r/name/'train/rank0.jsonl').read_text().splitlines()));last=rows[-1]
        progress[name]=dict(step=last['new_stage_step'],seconds=last['seconds'],loss=last['loss'],source_sha256=source_hash())
    report=dict(preflight_completed=True,experiment_completed=False,snapshot_only=True,source_sha256=source_hash(),parent_sha256=e['parent_sha256'],
        tests=counts,cpu_unique=len(cpu_ids),parities=eq['parities'],pilots=eq['pilots'],probes=probes,training_snapshot=progress,
        note='Training and future full validation are not complete. Never report this snapshot as final model accuracy.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    files={}
    for sub in ('src','tools','tests','configs'):
        for p in sorted((root/sub).rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:files[str(p.relative_to(root))]=p
    for directory in (r,root/'runs/target_audit',out):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.pth','.pyc'):files[str(p.relative_to(root))]=p
    for p in (root/'runs').iterdir():
        if p.is_file() and p.suffix in ('.log','.xml','.json'):files[str(p.relative_to(root))]=p
    files['pyproject.toml']=root/'pyproject.toml'
    archive=root/'runs/preflight_snapshot.tar.gz';assert not archive.exists();hashes={}
    with tarfile.open(archive,'w:gz') as tar:
        for name,p in files.items():
            raw=p.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest();m=tarfile.TarInfo(name);m.size=len(raw);tar.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(snapshot_only=True,experiment_completed=False,files=hashes),indent=2).encode();m=tarfile.TarInfo('SHA256.json');m.size=len(raw);tar.addfile(m,io.BytesIO(raw))
    receipt=dict(snapshot_only=True,experiment_completed=False,archive=str(archive),sha256=sha(archive),files=len(hashes),bytes=archive.stat().st_size)
    (out/'archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
