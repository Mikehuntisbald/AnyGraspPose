"""CPU audit of paired initialization and effective parameter movement."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch


def digest(value):
    h=hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode())
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=repr):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    baseline=None;records={}
    for arm in ('control','priority'):
        root=a.root/arm/'runs/seed42';start=torch.load(root/'stage_start.pt',map_location='cpu',weights_only=False)
        hashes={key:digest(start[key]) for key in ('model','optimizer','scheduler','rng')}
        if baseline is None:baseline=dict(hashes)
        assert hashes==baseline,'Paired arms did not share full starting state'
        config=start['config'];config['paths']['output']='ARM_OUTPUT';config['geometry_priority']['enabled']='ARM_FLAG'
        hashes['normalized_config']=digest(config)
        if records:assert hashes['normalized_config']==records['control']['hashes']['normalized_config']
        row=dict(hashes=hashes,source_step=start['step'],world=len(start['rng']))
        if (root/'last.pt').exists():
            last=torch.load(root/'last.pt',map_location='cpu',weights_only=False);groups={}
            for name,value in last['model'].items():
                if not value.is_floating_point() or name.startswith('ema_teacher.'):continue
                group='surface' if name.startswith('surface_head.') else ('encoder' if name.startswith('encoder.') else ('pose' if name=='query' or name.startswith(('head.','object_attn.','object_norm.','geometry_readout.')) else 'jepa_and_inputs'))
                original=start['model'][name];g=groups.setdefault(group,dict(delta_squared=0.,parent_squared=0.,elements=0))
                g['delta_squared']+=float((value.double()-original.double()).square().sum());g['parent_squared']+=float(original.double().square().sum());g['elements']+=value.numel()
            row.update(step=last['step'],parameter_movement={k:dict(delta_norm=v['delta_squared']**.5,relative_norm=(v['delta_squared']/max(v['parent_squared'],1e-30))**.5,elements=v['elements']) for k,v in groups.items()})
            del last
        records[arm]=row;del start
    result=dict(completed=True,full_starting_model_adam_scheduler_rng_identical=True,only_declared_config_differences=True,arms=records)
    (a.root/'paired_state_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
