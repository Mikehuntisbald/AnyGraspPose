"""Deterministic, rank-shared history curriculum in optimizer-step units."""
import numpy as np
MODES=('noisy_gt','lip_only','lip_fp')
def probabilities(config,step):
    start=config['fpaware_start_step'];end=config['fpaware_end_step']
    if end<=start:raise ValueError('Curriculum end must follow start')
    alpha=float(np.clip((step-start)/(end-start),0,1))
    a=np.array(config['fpaware_initial_probs'],dtype=float);b=np.array(config['fpaware_final_probs'],dtype=float)
    for p in [a,b]:
        if p.shape!=(3,) or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError('Invalid history probabilities')
    return (a*(1-alpha)+b*alpha).tolist()
def choose(config,step):
    p=probabilities(config,step)
    rng=np.random.default_rng(config['seed']+step)
    return str(rng.choice(MODES,p=p)),p
