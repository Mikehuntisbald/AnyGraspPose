"""Bounded geometry diagnostic with strict resume and paired input probes."""
import json,os,subprocess,sys,time,signal,hashlib,tarfile
import argparse
import yaml
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_transport_warmup_v35')
    parser.add_argument('--config',default='configs/jepa/geometry_transport_warmup_v35.yaml')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--clean-probe',action='store_true')
    args=parser.parse_args()
    exe=Path(__file__).resolve().parents[1]
    root=Path(args.root);root.mkdir(exist_ok=False)
    config=args.config;out=root/'runs/seed42'
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children=[]
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[x.pid for x in children],time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands,sharded=False):
        nonlocal children
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items())
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                log=(root/f'{stage}.{rank}.log').open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(3)
            if any(p.returncode for p in children):raise RuntimeError(stage+' failed')
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for log in logs:log.close()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        checks=['tests/jepa/test_cad_transport.py','tests/jepa/test_canonical_surface_targets.py','tests/jepa/test_execution_speed.py','tests/jepa/test_paired_geometry_curriculum.py']
        if yaml.safe_load((exe/config).read_text()).get('supervision_quality',{}).get('enabled'):
            checks.append('tests/jepa/test_supervision_quality.py')
        if yaml.safe_load((exe/config).read_text()).get('cad_atlas',{}).get('enabled'):
            checks.append('tests/jepa/test_cad_atlas_decoder.py')
        if yaml.safe_load((exe/config).read_text()).get('cad_image',{}).get('enabled'):
            checks.append('tests/jepa/test_cad_image_correspondence.py')
        run('tests',[['-m','pytest','-q',*checks]])
        for step in (2,args.steps):
            command=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_geometry_transport.py','--config',config,'--stop-at',str(step)]
            if step!=2:command+=['--resume',str(out/'last.pt')]
            run(f'train{step}',[command])
            probe_step=0 if step==2 else step
            ck=out/('initial.pt' if probe_step==0 else 'last.pt')
            run(f'probe{probe_step}',[['tools/probe_geometry_transport.py','--config',config,'--checkpoint',str(ck),'--out',str(root/'probe'/f'step{probe_step}'/f'rank{i}'),'--rank',str(i),'--world','8','--records','8','--lookup-audit'] for i in range(8)],True)
            if args.clean_probe:
                run(f'clean_probe{probe_step}',[['tools/probe_geometry_transport.py','--config',config,'--checkpoint',str(ck),'--out',str(root/'probe'/f'step{probe_step}_clean'/f'rank{i}'),'--rank',str(i),'--world','8','--records','8','--lookup-audit','--clean-control'] for i in range(8)],True)
        c=yaml.safe_load((exe/config).read_text())
        status('complete',completed=True,updates=c['geometry_transport_training']['updates'],backbone_frozen=c['geometry_transport_training'].get('decoder_only',False),default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
