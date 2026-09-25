import torch
from torch import nn
from lip.unified.serial_completion import CompletionRelations,pack_completion
from lip.unified.shared_patch_joint import SharedPatchRelations,restore_with_patch_extension
from test_serial_completion import fixture


def test_zero_adapter_preserves_old_output_then_both_paths_receive_gradient():
    torch.manual_seed(42);old=CompletionRelations();new=SharedPatchRelations(1.)
    new.load_state_dict(old.state_dict(),strict=False)
    args=fixture();args['visibility'].fill_(-10)
    args['surface']=torch.randn_like(args['surface'])*.02;args['surface'].requires_grad_()
    args['mid'].requires_grad_();args['last']=args['last'].clone().requires_grad_()
    packet=pack_completion(**args,feature_source='decoded')
    patch=torch.randn(1,256,256,requires_grad=True);packet['patch_feature']=patch
    a=old(packet,args['base']);b=new(packet,args['base'])
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with torch.no_grad():new.patch_projection[-1].weight.normal_(std=.01)
    tokens,_,moments,_=new(packet,args['base']);(tokens[...,0].sum()+moments.sum()).backward()
    for value in (patch,args['mid'],args['last'],args['surface']):
        assert value.grad is not None and value.grad.isfinite().all() and value.grad.abs().sum()>0
    assert args['surface'].grad[:,:3].abs().sum()>0
    assert args['surface'].grad[:,3].abs().sum()>0


class Toy(nn.Module):
    def __init__(self,extension=False):
        super().__init__();self.core=nn.Linear(2,3);self.geometry_readout=SharedPatchRelations(1.) if extension else CompletionRelations()


def test_extension_preserves_named_adam_and_existing_model_exactly():
    old=Toy();pairs=list(old.named_parameters())
    opt=torch.optim.AdamW([dict(params=[p for n,p in pairs],names=[n for n,p in pairs])],lr=.01)
    sum(p.square().sum() for p in old.parameters()).backward();opt.step()
    parent=dict(model=old.state_dict(),optimizer=opt.state_dict())
    new=Toy(True);pairs=list(reversed(list(new.named_parameters())))
    newopt=torch.optim.AdamW([dict(params=[p for n,p in pairs],names=[n for n,p in pairs])],lr=.001)
    result=restore_with_patch_extension(new,newopt,parent)
    assert len(result['new_optimizer_parameters'])==4 and len(result['added_model_keys'])==4
    for key,value in parent['model'].items():torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)
    assert newopt.param_groups[0]['lr']==.001
    assert all(not newopt.state.get(p) for n,p in pairs if n.startswith('geometry_readout.patch_projection.'))
