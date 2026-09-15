"""Fetch the terminal spatial-factorial archive and publish only verified local results."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--host',required=True);p.add_argument('--port',required=True);p.add_argument('--remote-out',required=True)
    p.add_argument('--local-out',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    a.local_out.mkdir(parents=True,exist_ok=False);state=dict(phase='waiting',started=time.time(),remote_out=a.remote_out)
    def save():
        state['updated']=time.time();f=a.local_out/'status.tmp';f.write_text(json.dumps(state,indent=2));f.replace(a.local_out/'status.json')
    def remote(name):
        value=subprocess.check_output(['ssh','-p',a.port,'-o','BatchMode=yes','-o','ConnectTimeout=15',a.host,'cat '+shlex.quote(a.remote_out+'/'+name)],text=True,timeout=30)
        return json.loads(value)
    try:
        save();deadline=time.time()+8*3600
        while True:
            try:s=remote('status.json')
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                if not a.wait or time.time()>deadline:raise
                state['last_connection_error']=str(error);save();time.sleep(30);continue
            if s['phase']=='failed':raise RuntimeError('Remote delivery failed: '+str(s.get('error')))
            if s['phase']=='completed':break
            if not a.wait or time.time()>deadline:raise RuntimeError('Remote delivery not complete: '+s['phase'])
            state['remote_phase']=s['phase'];save();time.sleep(30)
        receipt=remote('archive.json');assert receipt['completed'];state['phase']='downloading';save()
        archive=a.local_out/'delivery.tar.gz'
        subprocess.run(['scp','-P',a.port,'-o','BatchMode=yes',a.host+':'+receipt['archive'],str(archive)],check=True,timeout=1800)
        assert archive.stat().st_size==receipt['bytes'] and digest(archive)==receipt['sha256']
        (a.local_out/'archive.json').write_text(json.dumps(receipt,indent=2));state['phase']='verifying';save()
        dest=a.local_out/'evidence';dest.mkdir()
        with tarfile.open(archive,'r:gz') as tar:tar.extractall(dest,filter='data')
        manifest=json.loads((dest/'SHA256.json').read_text());assert len(manifest['files'])==receipt['files']
        for name,expected in manifest['files'].items():assert digest(dest/name)==expected,name
        result=dest/'runs/completion/RESULT.md'
        if result.exists():shutil.copy2(result,a.local_out/'RESULT.md')
        verification=dict(completed=True,experiment_completed=True,archive_sha256=digest(archive),files_verified=len(manifest['files']),remote_out=a.remote_out)
        (a.local_out/'verification.json').write_text(json.dumps(verification,indent=2))
        root=Path(__file__).resolve().parents[1]
        report_text=result.read_text().replace('(figures/','(../runs/spatial_alignment_factorial_20260915_completed/evidence/runs/completion/figures/')
        report_text+='\n[本地归档校验](../runs/spatial_alignment_factorial_20260915_completed/verification.json)。\n'
        target=root/'docs/SPATIAL_ALIGNMENT_FACTORIAL_RESULTS_20260915.md'
        if target.exists():raise FileExistsError(target)
        target.write_text(report_text)
        ongoing=root/'docs/ONGOING_OCCLUSION_OPTIMIZATION.md'
        marker='## 2026-09-15：显式空间对齐监督与父 latent 的 2×2 完整验证完成'
        if marker not in ongoing.read_text():
            with ongoing.open('a') as f:f.write('\n\n'+marker+'\n\n四组各 1,000 步、各 320 流 / 23,200 帧，配对分析及独立测速均完成。\n终态证据已自动回传，'+str(verification['files_verified'])+' 个文件逐个验 SHA。没有自动晋级或新 official test。\n\n[完整结果与配对区间](SPATIAL_ALIGNMENT_FACTORIAL_RESULTS_20260915.md)。\n')
        state.update(phase='completed',completed=time.time());save();print(json.dumps(verification),flush=True)
    except BaseException as error:
        state.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
