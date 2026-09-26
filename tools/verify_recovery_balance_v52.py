"""CPU checkpoint receipt: matched initialization, fixed pose, complete budgets."""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.reconstruction_only import is_pose_parameter


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    initial=torch.load(a.root/'control/runs/seed42/initial.pt',map_location='cpu',weights_only=False)
    other=torch.load(a.root/'balanced/runs/seed42/initial.pt',map_location='cpu',weights_only=False)
    assert initial['step']==other['step']==0
    assert initial['model'].keys()==other['model'].keys()
    assert all(torch.equal(v,other['model'][k]) for k,v in initial['model'].items())
    del other
    result=dict(verified=True,initial_model_tensors_equal=len(initial['model']),pose_training=False,arms={})
    for arm in ('control','balanced'):
        path=a.root/arm/'runs/seed42/last.pt'
        end=torch.load(path,map_location='cpu',weights_only=False)
        assert end['step']==200
        frozen=[k for k in initial['model'] if is_pose_parameter(k)]
        assert frozen and all(torch.equal(initial['model'][k],end['model'][k]) for k in frozen)
        provenance=json.loads((path.parent/'provenance.json').read_text())
        assert provenance['pose_loss'] is False and provenance['dino_loss'] is False
        resume=json.loads((path.parent/'resume2.json').read_text())
        assert resume['complete_state_verified']
        result['arms'][arm]=dict(terminal_sha256=sha(path),step=end['step'],pose_tensors_bitwise_unchanged=len(frozen),
            strict_resume_verified=True,source_checkpoint_sha256=provenance['source_checkpoint_sha256'])
        del end
    for rank in range(8):
        first=[json.loads((a.root/arm/f'runs/seed42/rank{rank}.jsonl').read_text().splitlines()[0]) for arm in ('control','balanced')]
        assert first[0]['metrics']==first[1]['metrics'],f'Initial forward differs at rank{rank}'
    result['initial_forward_metrics_equal_all_ranks']=True
    (a.root/'checkpoint_verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
