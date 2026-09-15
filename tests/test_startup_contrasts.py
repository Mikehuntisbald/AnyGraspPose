import numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_startup_factorial import contrasts


def test_factorial_contrasts_include_interaction_and_frozen_parent():
    # S effect 3, O effect 2, interaction 4; marginal main effects include half
    # the interaction. The external frozen parent is not part of the factorial.
    values=np.array([100.,0.,2.,3.,9.]);c=contrasts()
    assert values@c['S_main']==5 and values@c['O_main']==4 and values@c['interaction']==4
    assert values@c['S1O1_vs_parent']==-91 and values@c['S1O1_vs_S0O0']==9
    assert all(sum(v)==0 for v in c.values())


def test_previous_candidate_does_not_change_factorial_contrasts():
    values=np.array([100.,0.,2.,3.,9.,8.]);c=contrasts(True)
    assert values@c['S_main']==5 and values@c['O_main']==4 and values@c['interaction']==4
    assert values@c['S1O1_vs_previous_candidate']==1 and values@c['S0O0_vs_previous_candidate']==-8
    assert all(sum(v)==0 for v in c.values())
