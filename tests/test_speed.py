import copy
import torch
from lip.data.synthetic import synthetic_item
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.models.tracker import Tracker


def test_preloaded_rollout_preserves_predictions_losses_and_gradients():
    torch.set_num_threads(2);torch.manual_seed(71)
    item=synthetic_item(length=2,size=64)
    model=Tracker(False,dropout=0.).eval()
    torch.nn.init.normal_(model.head[-1].weight,std=.002)
    c=dict(clip_length=2,image_size=32,precision='fp32',crop_expansion=2.)
    original=copy.deepcopy(item)
    a,out_a=batch_step(model,[item],Renderer('cpu'),c,4,True,True)
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    b,out_b=batch_step(model,[item],Renderer('cpu'),dict(c,preload_rollout_observations=True),4,True,True)
    assert a['loss']==b['loss']
    for x,y in zip(out_a,out_b):assert torch.equal(x,y)
    for n,p in model.named_parameters():
        if n in grads:assert torch.equal(grads[n],p.grad)
    for key in ('rgb','depth','poses','k'):assert torch.equal(item[key],original[key])
