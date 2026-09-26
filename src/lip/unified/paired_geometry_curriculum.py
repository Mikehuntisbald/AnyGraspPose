"""Training-only paired pose errors, sharing an identical observed crop."""
from dataclasses import replace
import torch


@torch.no_grad()
def mirror_estimate(estimate,truth):
    """Inverse camera-frame rotation error and opposite center translation error."""
    result=estimate.clone()
    error=estimate[:,:3,:3]@truth[:,:3,:3].transpose(-1,-2)
    result[:,:3,:3]=error.transpose(-1,-2)@truth[:,:3,:3]
    result[:,:3,3]=2*truth[:,:3,3]-estimate[:,:3,3]
    return result


@torch.no_grad()
def paired_scenes(scenes,truth,renderer):
    bases=mirror_estimate(torch.stack([s.pose for s in scenes]),truth)
    result=[]
    for scene,base in zip(scenes,bases):
        state=scene.state.clone();state[:6]=base[:3,:2].T.flatten();state[6:9]=base[:3,3]/scene.diameter
        result.append(replace(scene,pose=base,state=state,render=renderer(scene.cad['appearance'],base,scene.k_crop,224)))
    return result
