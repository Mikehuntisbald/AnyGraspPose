"""Parent preservation for a registered directly supervised residual tracker."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.stream_checkpoint import source_hash
from lip.engine.stream_training import StreamTrainingModule
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args()
    e=json.loads((a.experiment/'experiment.json').read_text());torch.set_num_threads(2)
    ds=StreamClips(e['data_root'],e['index_root'],8,48,fixed=json.loads(Path(e['training_manifest']).read_text()));samples=[ds[i] for i in range(2)]
    renderer=Renderer('cuda');models={};configs={};results={}
    for name in ('control','direct_pose'):
        c=load_stream_config(e['arms'][name]['config']);model=make_model(c).cuda().eval()
        model.load_state_dict(torch.load(e['arms'][name]['init'],map_location='cpu',weights_only=False)['model']);models[name]=model;configs[name]=c
    for precision in ('fp32','bf16'):
        predictions={}
        for name,model in models.items():
            with torch.no_grad():predictions[name]=StreamTrainingModule(model,dict(configs[name],precision=precision),renderer)(samples,True)['predictions']
        assert torch.equal(predictions['control'],predictions['direct_pose'])
        results[precision]=dict(bitwise_equal=True,frames=112)
    (a.experiment/'equivalence.json').write_text(json.dumps(dict(passed=True,parent_sha256=e['parent_sha256'],source_sha256=source_hash(),
        scope='Registered direct-pose architecture on two identical actual training fragments with configured augmentation; zero-residual function preservation',results=results),indent=2))


if __name__=='__main__':main()
