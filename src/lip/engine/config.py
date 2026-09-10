import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import torch
import yaml


def load_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c['precision']=='bf16' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('bf16 requires a supported CUDA GPU; choose explicit precision: fp32 for CPU correctness tests')
    return c


def environment():
    import importlib.metadata as md
    versions={}
    for n in ['torch','torchvision','numpy','scipy','trimesh','opencv-python-headless','PyYAML','pytest','nvdiffrast','tensorboard']:
        try:versions[n]=md.version(n)
        except md.PackageNotFoundError:versions[n]=None
    gpu=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i);gpu.append(dict(index=i,name=p.name,total_bytes=p.total_memory))
    limits={}
    for name,path in [('cpu_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),('cpu_period_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us'),('memory_limit_bytes','/sys/fs/cgroup/memory/memory.limit_in_bytes')]:
        if Path(path).exists():limits[name]=int(Path(path).read_text())
    return dict(versions=versions,cuda_runtime=torch.version.cuda,gpus=gpu,cgroup_limits=limits,
                driver=subprocess.getoutput('nvidia-smi --query-gpu=driver_version --format=csv,noheader').splitlines(),
                cpu_affinity=len(os.sched_getaffinity(0)),cpu_count=os.cpu_count(),disk=shutil.disk_usage('.')._asdict(),
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def optimizer_and_scheduler(model,c):
    groups={}
    for name,p in model.named_parameters():
        rgb=name.startswith('rgb.');decay=p.ndim>1 and not name.endswith('bias')
        groups.setdefault((rgb,decay),[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=params,lr=c['lr_rgb_backbone'] if rgb else c['lr_new_modules'],
                                    weight_decay=c['weight_decay'] if decay else 0.,rgb=rgb)
                                for (rgb,decay),params in groups.items()])
    def factor(step,rgb):
        warm=c['warmup_optimizer_steps'];maxstep=c['max_optimizer_steps']
        if rgb and step<warm:return 0.
        if not rgb and step<warm:return (step+1)/max(1,warm)
        ratio=min(1,max(0,(step-warm)/max(1,maxstep-warm)))
        decay=c['min_lr_ratio']+(1-c['min_lr_ratio'])*.5*(1+math.cos(math.pi*ratio))
        return decay*(min(1,(step-warm+1)/max(1,c.get('backbone_ramp_steps',1000))) if rgb else 1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[lambda s,rgb=g['rgb']:factor(s,rgb) for g in optimizer.param_groups])
    return optimizer,scheduler


def check_data_gate(index, allow_verified_subset=False):
    index=Path(index)
    if not (index/'audit.json').exists():raise RuntimeError(f'Official data index is absent at {index}; provide extracted DEX_YCB_DIR and run scripts/preflight.sh')
    audit=json.loads((index/'audit.json').read_text())
    complete=audit.get('complete') or (allow_verified_subset and (audit.get('verified_train_subset') or audit.get('verified_subject_subset')) and not audit.get('errors'))
    if not complete or not audit.get('split_disjoint'):raise RuntimeError('Incomplete/unverified official split')
    gate=json.loads((index/'geometry_gate.json').read_text())
    if not gate.get('passed') or gate.get('split_hash')!=audit['split_hash'] or gate.get('mesh_hash')!=audit['mesh_hash']:
        raise RuntimeError('Geometry gate missing, failed or stale')
    return audit
