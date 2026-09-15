"""Audit and archive this matched experiment; distinguish snapshots from completion."""
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
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--fp-runtime',type=Path,required=True);p.add_argument('--posecnn-runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wait-completion',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    state=dict(phase='starting',started=time.time(),completion_required=a.wait_completion)
    def save():
        state['updated']=time.time();tmp=a.out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(a.out/'status.json')
    try:
        save();e=json.loads((a.experiment/'experiment.json').read_text());assert source_hash()==e['source_sha256']
        if a.wait_completion:
            state['phase']='waiting';save()
            while True:
                states=[json.loads((p/'status.json').read_text()) for p in (a.experiment,a.evaluation)]
                if any(s['phase']=='failed' for s in states):raise RuntimeError('Upstream failure: '+repr(states))
                if all(s['phase']=='completed' for s in states):break
                save();time.sleep(15)
        for path,key in [(e['parent'],'parent_sha256'),(e['train_initializers'],'train_initializers_sha256'),(e['val_initializers'],'val_initializers_sha256'),(e['training_manifest'],'training_manifest_sha256')]:assert sha(path)==e[key]
        proofs={}
        for label,path in [('training_cpu',a.experiment/'tests.xml'),('training_cuda',a.experiment/'cuda_tests.xml'),('scorecard',root/'runs/scorecard_tests.xml'),('fp_protocol',a.fp_runtime/'runs/tests.xml')]:
            suites=list(ET.parse(path).getroot().iter('testsuite'));counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
            assert counts['failures']==counts['errors']==0;proofs[label]=dict(path=str(path),sha256=sha(path),**counts)
        equivalent=json.loads((a.experiment/'equivalence.json').read_text());assert equivalent['passed'] and equivalent['all_trainable_gradients_finite']
        reference=a.fp_runtime/'runs/full_v2/scored';m=json.loads((reference/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['fp_calls']==22303 and m['missing_pose_frames']==578 and sha(reference/'predictions.jsonl')==m['predictions_sha256']
        report=dict(completed=a.wait_completion,snapshot_only=not a.wait_completion,tests=proofs,source_sha256=source_hash(),parent_sha256=e['parent_sha256'],
            train_initializers_sha256=e['train_initializers_sha256'],val_initializers_sha256=e['val_initializers_sha256'],
            fp_baseline_complete=True,real_training_interface_passed=True,training_status=json.loads((a.experiment/'status.json').read_text()),
            evaluation_status=json.loads((a.evaluation/'status.json').read_text()),test_launched=False)
        if a.wait_completion:
            selection=json.loads((a.evaluation/'selection.json').read_text());assert selection['completed'] and not selection['test_launched']
            assert sha(a.evaluation/'selected.pt')==selection['checkpoint_sha256'];comparison=json.loads((a.evaluation/'comparison/comparison.json').read_text())
            assert comparison['bootstrap']['physical_sequences']==40
            for name in ('control','real_mix'):
                receipt=json.loads((a.experiment/name/'training_receipt.json').read_text());assert receipt['completed'] and receipt['stage_step']==1000
                assert receipt['checkpoint_sha256']==comparison['checkpoints'][name]==sha(a.experiment/name/'train/last.pt')
            for name in ('residual','control','real_mix'):
                folder=a.evaluation/name/'scored';m=json.loads((folder/'manifest.json').read_text())
                assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
                assert m['fp_calls']==m['gt_pose_reads']==m['gt_mask_reads']==m['gt_resets']==0 and m['missing_pose_frames']==578
                assert m['initializers_sha256']==e['val_initializers_sha256'] and sha(folder/'predictions.jsonl')==m['predictions_sha256']
            for name in dict.fromkeys(('fp','residual',selection['selected'])):
                folder=a.evaluation/'benchmark'/name;bm=json.loads((folder/'receipt.json').read_text());assert bm['completed']
                assert bm['fp_calls']==(640 if name=='fp' else 0) and bm['metrics']['all_updates']['frames']==640
                assert bm['metrics']['steady_after8']['frames']==480 and sha(folder/'frames.jsonl')==bm['frames_sha256']
            with (a.out/'figures.log').open('w') as f:
                subprocess.run([sys.executable,'tools/plot_fp_scorecard.py','--comparison',str(a.evaluation/'comparison'),'--out',str(a.out/'figures')],cwd=root,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,check=True)
            report['selection']=selection
            text=(a.evaluation/'comparison/report.md').read_text()
            text+='\n## Isolated latency (CPU RGB-D to CPU pose, no IO)\n\n| Method | Median (ms) | P95 (ms) |\n|---|---:|---:|\n'
            for name,values in selection['timing'].items():
                v=values['steady_after8'];text+=f"| {name} | {v['p50_ms']:.3f} | {v['p95_ms']:.3f} |\n"
            text+='\nAll entries, paired intervals, failure populations and memory limitations remain in the machine-readable receipts. This is one training seed and native val, not official test AR.\n'
            (a.out/'RESULT.md').write_text(text)
        (a.out/'verification.json').write_text(json.dumps(report,indent=2));state['phase']='packaging';save()
        inputs={}
        for label,folder in [('experiment',root),('fp',a.fp_runtime),('posecnn_train',a.posecnn_runtime)]:
            for sub in ('src','tools','tests','docs','runs'):
                for path in sorted((folder/sub).rglob('*')):
                    if path.is_file() and path.suffix not in ('.pt','.pth','.pyc','.so','.gz','.tar','.zip') and '__pycache__' not in path.parts:
                        inputs[label+'/'+str(path.relative_to(folder))]=path
            if (folder/'pyproject.toml').exists():inputs[label+'/pyproject.toml']=folder/'pyproject.toml'
        # Archived independent initializer and retained LIP outputs are bound and
        # included, so the matched-baseline comparison is usable off the server.
        initial=Path(e['val_initializers']);inputs['val_initializers/initializers.json']=initial
        original=Path(json.loads((a.evaluation/'spec.json').read_text())['args']['residual_scored'])
        for name in ('manifest.json','metrics.json','predictions.jsonl'):inputs['prior_residual/'+name]=original/name
        archive=a.out.parent/(a.out.name+'.tar.gz');digests={}
        with tarfile.open(archive,'w:gz') as tar:
            for name,path in inputs.items():
                raw=path.read_bytes();digests[name]=hashlib.sha256(raw).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
            raw=json.dumps(dict(files=digests,snapshot_only=not a.wait_completion,scope='Captured file bytes; weights remain on the server and are SHA-bound. A running snapshot is not experiment completion.'),indent=2).encode()
            entry=tarfile.TarInfo('SHA256.json');entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
        receipt=dict(completed=True,experiment_completed=a.wait_completion,path=str(archive),sha256=sha(archive),files=len(digests),bytes=archive.stat().st_size)
        (a.out/'archive.json').write_text(json.dumps(receipt,indent=2));state.update(phase='completed',completed=time.time());save();print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
