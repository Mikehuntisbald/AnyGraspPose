"""Visualize train-only correspondence construction on real RGB-D; no optimization."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips
from lip.engine.stream_features import build_current_features
from lip.geometry.renderer import Renderer
from lip.losses_spatial_alignment import correspondence_targets,sample_nearest
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    root=Path(__file__).resolve().parents[1];e=json.loads((root/'runs/factorial/experiment.json').read_text());out=root/'runs/target_audit';out.mkdir(exist_ok=False)
    cohort=json.loads(Path('/mnt/why/dexycb_lip/alignment_diagnosis_20260915/runs/train_diagnostic/spec.json').read_text())['cohort']
    selected=[cohort['fit'][0],cohort['fit'][2],cohort['probe'][7],cohort['fit'][6]]
    torch.set_num_threads(2)
    data=StreamClips(e['data_root'],e['index_root'],0,1,fixed=json.loads(Path(e['training_manifest']).read_text()),decode_threads=0,
        external_initializers=e['train_initializers'],external_initializers_sha256=e['train_initializers_sha256'],real_initialization_probability=.5,include_initial_observation=True)
    renderer=Renderer('cpu');fig,axes=plt.subplots(4,3,figsize=(10,12),constrained_layout=True);receipt=[]
    for i,item in enumerate(selected):
        s=data[item['manifest_index']];rgb=s['rgb'][0].float()/255;depth=s['depth'][0].float()*s['depth_scale'];base=s['real_initial_pose']
        features,diag=build_current_features(rgb,depth,base,s['k'],s['mesh'],renderer,s['timestamps'][1],s['timestamps'][0],size=224)
        gt_depth,_=renderer(s['mesh'],s['targets'][0],diag['K_crop'],224)
        eligible=torch.tensor([s['stream']['object_id'] not in e['symmetric_object_ids']])
        distribution,supported,coverage=correspondence_targets(features['geometry'][None],torch.tensor([float(s['mesh']['diameter'])]),s['targets'][0,None],
            diag['K_crop'][None],diag['A'][None],depth[None],gt_depth[None],eligible)
        crop=diag['crop_rgb'].permute(1,2,0).numpy().clip(0,1)
        axes[i,0].imshow(crop);axes[i,1].imshow(diag['render_depth'][0],cmap='viridis');axes[i,2].imshow(crop)
        key_supported=distribution[0].sum(0)>0;keys=key_supported.nonzero()[:,0]
        for key in keys[::max(1,len(keys)//18)].tolist():
            query=int(distribution[0,:,key].argmax());sx=(key%14+.5)*16-.5;sy=(key//14+.5)*16-.5
            tx=(query%14+.5)*16-.5;ty=(query//14+.5)*16-.5
            axes[i,2].plot([sx,tx],[sy,ty],'-',color='cyan',lw=.8);axes[i,2].plot(tx,ty,'.',color='red',ms=4)
        for ax in axes[i]:ax.set_xlim(0,223);ax.set_ylim(223,0);ax.axis('off')
        axes[i,0].set_title(f"Train object {s['stream']['object_id']} / init {item['initial_rotation_deg']:.1f} deg")
        axes[i,1].set_title('CAD rendered at real prior')
        axes[i,2].set_title(f"Visible targets: {len(keys)} keys / {int(supported.sum())} queries" if bool(eligible[0]) else 'Symmetric: auxiliary excluded')
        receipt.append(dict(sample=item,eligible=bool(eligible[0]),visible_keys=int(coverage['visible_keys']),supported_queries=int(supported.sum()),
            row_sum_max_error=float((distribution.sum(-1)[supported]-1).abs().max()) if supported.any() else None))
    fig.suptitle('GT used only to construct TRAIN targets; cyan source-to-target token displacement, red target cell',fontsize=11)
    fig.savefig(out/'targets.png',dpi=170);fig.savefig(out/'targets.pdf');plt.close(fig)
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,source_sha256=source_hash(),entrypoint_sha256=sha(Path(__file__)),
        split='train',renderer='CPU audit; formal training uses validated CUDA renderer',augmentation='none in this visual geometry audit; formal training retains S1O1',records=receipt),indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
