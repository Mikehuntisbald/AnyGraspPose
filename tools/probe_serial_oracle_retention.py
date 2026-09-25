"""CPU-only frozen readout retention on the fixed train-development partition."""
import argparse,json,sys,time
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.runtime)/'src'));sys.path.insert(0,str(Path(a.runtime)/'tools'))
    import train_serial_oracle as module
    from lip.unified.reconstruction_only import is_pose_parameter
    from lip.engine.jepa_checkpoint import sha
    torch.set_num_threads(4);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=module.Readout();model.load_state_dict({k:v for k,v in saved['model'].items() if is_pose_parameter(k)});model.eval()
    step=saved['step'];del saved;records=[]
    for f in sorted(Path(a.cache).glob('rank*/packets.pt')):
        records += [x for x in torch.load(f,weights_only=False) if x['split']=='holdout']
    original=module.batch;module.batch=lambda records,ids:original(records,ids,device='cpu')
    start=time.monotonic();ideal=module.evaluate(model,records)
    for item in records:item['oracle']['feature']=item['predicted']['feature']
    predicted=module.evaluate(model,records)
    result=dict(step=step,checkpoint_sha256=sha(a.checkpoint),oracle_appearance=ideal,cached_student_appearance=predicted,
        seconds=time.monotonic()-start,precision='CPU FP32',official_test_access=False,optimizer_updates=0,
        scope='Both use ideal geometry; development sequences excluded from readout training/rehearsal. Not native accuracy.')
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
