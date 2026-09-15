"""Refine one observation repeatedly; replace its speculative cache, never append twice."""
import torch


@torch.no_grad()
def step_iterated(model,rgb,depth,timestamp,state,*,iterations=1,**kwargs):
    if type(iterations) is not int or iterations not in (1,2):raise ValueError('Use one or two iterations')
    if 'refinement_base' in kwargs:raise ValueError('Refinement base is owned by this transaction')
    if iterations==2 and (model.cache_kind!='functional' or model.architecture_id!='stream_dual_cross_residual'):
        raise ValueError('Two-pass v1 requires residual functional history')
    proposal,pending=model.step(rgb,depth,timestamp,state,**kwargs)
    attempted=1;accepted=int(proposal['status']=='ok');inner_failure=None
    if iterations==2 and accepted:
        attempted+=1
        second,next_pending=model.step(rgb,depth,timestamp,state,refinement_base=proposal['pose_centered'],**kwargs)
        if second['status']=='ok':proposal,pending=second,next_pending;accepted+=1
        else:inner_failure=second['status']
    proposal=dict(proposal,inner_iterations_requested=iterations,inner_iterations_attempted=attempted,
        inner_iterations_accepted=accepted,inner_failure=inner_failure)
    # pending still owns the selected tensor; caller commits exactly once.
    return proposal,pending
