"""Verify terminal training/screen receipts and package reproducible evidence."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--spec',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];spec=json.loads(a.spec.read_text());screen=Path(spec['out']);training=Path(spec['training_experiment'])
    a.out.mkdir(parents=True,exist_ok=False);status=dict(phase='waiting',started=time.time())
    def save():
        status['updated']=time.time();temp=a.out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(a.out/'status.json')
    try:
        save()
        while True:
            phases=[json.loads((p/'status.json').read_text()) for p in (screen,training)]
            if any(s['phase']=='failed' for s in phases):raise RuntimeError('Upstream failure: '+repr(phases))
            if all(s['phase']=='completed' for s in phases):break
            save();time.sleep(15)
        status['phase']='verifying';save()
        assert source_hash()==spec['source_sha256']
        for path,digest in spec['bound_files'].items():assert sha(path)==digest,path
        selection=json.loads((screen/'selection.json').read_text());assert selection['completed'] and not selection['test_launched']
        assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
        comparison=json.loads((screen/'comparison/comparison.json').read_text());assert comparison['completed'] and comparison['streams']==320 and comparison['frames']==23200
        assert comparison['bootstrap']['physical_sequences']==40
        proofs={}
        for label,path in [('training_cpu',training/'tests.xml'),('training_cuda',training/'cuda_tests.xml'),('non_gt_protocol',root/'runs/tests.xml')]:
            suites=list(ET.parse(path).getroot().iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
            assert counts['failures']==counts['errors']==0;proofs[label]=dict(path=str(path),sha256=sha(path),**counts)
        for name in ('control','smooth_rotation'):
            receipt=json.loads((training/name/'training_receipt.json').read_text());assert receipt['completed'] and receipt['stage_step']==1000
            assert receipt['checkpoint_sha256']==comparison['checkpoints'][name]
        for name,digest in comparison['checkpoints'].items():
            m=json.loads((screen/name/'scored/manifest.json').read_text());metric=json.loads((screen/name/'scored/metrics.json').read_text())
            assert m['checkpoint_sha256']==digest and sha(screen/name/'frozen.pt')==digest
            assert sha(screen/name/'scored/predictions.jsonl')==m['predictions_sha256']
            assert m['completed'] and m['population_verified'] and m['fp_calls']==m['gt_pose_reads']==m['gt_mask_reads']==0
            assert m['initializers_sha256']==comparison['initializers_sha256'] and m['frames']==23200 and len(m['streams'])==320
            assert metric['populations']['all']['missing_pose_frames']==578
        with (a.out/'figures.log').open('w') as log:
            subprocess.run([sys.executable,'tools/plot_val_non_gt.py','--run',str(screen),'--out',str(a.out/'figures')],cwd=root,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True)
        report=dict(completed=True,tests=proofs,source_sha256=source_hash(),screen_spec_sha256=sha(a.spec),
            all_five_full_val_runs_verified=True,models=list(comparison['checkpoints']),selected=selection,
            new_test_launched=False,initializers=319,delayed_streams=14,missing_streams=1,missing_pose_frames=578,
            scope='Fixed-budget single-seed matched retraining and real-initialized full native s0 val. Not new official test or SOTA evidence.')
        (a.out/'verification.json').write_text(json.dumps(report,indent=2));status['phase']='packaging';save()
        inputs={}
        for label,folder in [('evaluation',root),('training',training.parents[1]),('posecnn',Path(spec['initializers']).parents[2])]:
            for sub in ('src','tools','tests','docs','runs'):
                for path in sorted((folder/sub).rglob('*')):
                    if path.is_file() and path.suffix not in ('.pt','.pth','.pyc','.so') and '__pycache__' not in path.parts:
                        inputs[label+'/'+str(path.relative_to(folder))]=path
        for path in sorted(a.out.rglob('*')):
            if path.is_file():inputs['completion/'+str(path.relative_to(a.out))]=path
        archive=a.out.parent/(a.out.name+'.tar.gz');digests={}
        with tarfile.open(archive,'w:gz') as tar:
            for name,path in inputs.items():
                data=path.read_bytes();digests[name]=hashlib.sha256(data).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
            data=json.dumps(dict(files=digests,scope='Exact byte snapshots. Checkpoint binaries remain on the remote server and are identified by SHA in receipts.'),indent=2).encode()
            entry=tarfile.TarInfo('SHA256.json');entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
        receipt=dict(completed=True,path=str(archive),sha256=sha(archive),files=len(digests),bytes=archive.stat().st_size)
        (a.out/'archive.json').write_text(json.dumps(receipt,indent=2));status.update(phase='completed',completed=time.time());save();print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
