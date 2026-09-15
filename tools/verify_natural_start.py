"""Verify identical parent functions and exercise real natural hard fragments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.data.stream_clips import StreamClips
from lip.engine.stream_checkpoint import source_hash, sha
from lip.engine.stream_config import load_stream_config, make_model
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args()
    e=json.loads((a.experiment/'experiment.json').read_text());torch.set_num_threads(2)
    manifest=json.loads(Path(e['arms']['natural']['training_manifest']).read_text())
    selections=list(map(json.loads,Path(e['sampling_selections']).read_text().splitlines()))
    indices=[i for i,s in enumerate(selections) if s['hard_selected']][:2]
    assert len(indices)==2
    ds=StreamClips(e['data_root'],e['index_root'],8,48,fixed=manifest)
    samples=[ds[i] for i in indices];renderer=Renderer('cuda');models={};configs={};results={}
    fingerprints=[hashlib.sha256(s['rgb'].numpy().tobytes()+s['depth'].numpy().tobytes()).hexdigest() for s in samples]
    for name in ('control','natural'):
        c=load_stream_config(e['arms'][name]['config']);m=make_model(c).cuda().eval()
        m.load_state_dict(torch.load(e['arms'][name]['init'],map_location='cpu',weights_only=False)['model'])
        models[name]=m;configs[name]=c
    for precision in ('fp32','bf16'):
        values={}
        for name,m in models.items():
            with torch.no_grad():values[name]=StreamTrainingModule(m,dict(configs[name],precision=precision),renderer)(samples,True)
        x,y=values['control']['predictions'],values['natural']['predictions']
        assert torch.equal(x,y) and torch.isfinite(x).all() and torch.isfinite(values['natural']['loss'])
        results[precision]=dict(bitwise_equal=True,frames=x.shape[0]*x.shape[1],max_pose_difference=float((x-y).abs().max()),loss=float(values['natural']['loss']))
    assert fingerprints==[hashlib.sha256(s['rgb'].numpy().tobytes()+s['depth'].numpy().tobytes()).hexdigest() for s in samples]
    (a.experiment/'equivalence.json').write_text(json.dumps(dict(passed=True,parent_sha256=e['parent_sha256'],source_sha256=source_hash(),
        scope='Both identical initial models consume the SAME two actual natural-hard fragments with configured augmentation. Different formal sampling manifests are not expected to produce identical losses.',
        sample_indices=indices,samples=[s['sample'] for s in samples],raw_tensors_unchanged=True,results=results,
        init_sha256={name:sha(e['arms'][name]['init']) for name in models}),indent=2))


if __name__=='__main__':main()
