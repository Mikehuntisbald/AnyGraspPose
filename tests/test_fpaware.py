import torch,numpy as np
from lip.engine.rollout import probabilities,choose
from lip.engine.runtime import batch_step
from lip.data.synthetic import synthetic_item
from lip.models.tracker import Tracker
from lip.geometry.renderer import Renderer

def test_curriculum_endpoints_and_determinism():
 c=dict(seed=42,fpaware_start_step=19000,fpaware_end_step=40000,fpaware_initial_probs=[.4,.3,.3],fpaware_final_probs=[.2,.2,.6])
 assert np.allclose(probabilities(c,19000),[.4,.3,.3]);assert np.allclose(probabilities(c,40000),[.2,.2,.6])
 assert np.allclose(probabilities(c,29500),[.3,.25,.45])
 assert choose(c,19001)==choose(c,19001)

def test_fp_transition_detached_and_loss_on_lip_no_future_gt():
 torch.set_num_threads(2);torch.manual_seed(42)
 item=synthetic_item(length=2,size=64);item['stream']={'mesh_path':'synthetic'}
 model=Tracker(False,dropout=0.).eval();torch.nn.init.normal_(model.head[-1].weight,std=.002)
 c=dict(clip_length=2,image_size=32,precision='fp32',crop_expansion=2.)
 env_weight=torch.nn.Parameter(torch.tensor(.01));accepted=[];inputs=[]
 def env(pred,rgb,depth,k,path,center):
  assert not torch.is_grad_enabled() and not pred.requires_grad
  result=pred.clone();result[0,3]+=env_weight;accepted.append(result.detach().clone());return result
 hook=model.register_forward_pre_hook(lambda m,a,k:inputs.append(k['T_base_centered'].detach().clone()),with_kwargs=True)
 vals,out=batch_step(model,[item],Renderer('cpu'),c,4,False,True,history_mode='lip_fp',fp_transition=env)
 assert len(accepted)==3 and env_weight.grad is None
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
 for u in range(1,4):assert torch.equal(inputs[u][0],accepted[u-1])
 item['poses'][2:,:3,3]+=.2
 _,out2=batch_step(model,[item],Renderer('cpu'),c,4,False,False,history_mode='lip_fp',fp_transition=env)
 for a,b in zip(out,out2):assert torch.equal(a,b)
 hook.remove()

def test_noisy_gt_refreshes_strict_past_only():
 torch.set_num_threads(2)
 item=synthetic_item(length=2,size=64);model=Tracker(False,dropout=0.).eval()
 c=dict(clip_length=2,image_size=32,precision='fp32',crop_expansion=2.)
 bases=[];hook=model.register_forward_pre_hook(lambda m,a,k:bases.append(k['T_base_centered'].clone()),with_kwargs=True)
 batch_step(model,[item],Renderer('cpu'),c,4,False,False,history_mode='noisy_gt')
 for u,b in enumerate(bases):assert torch.equal(b[0],item['poses'][u+1])
 hook.remove()
