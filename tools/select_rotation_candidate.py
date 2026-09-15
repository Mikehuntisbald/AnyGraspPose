"""Apply a pre-test conservative validation rule; retain the fixed recipe on failure."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_checks(comparison,bad_initial,robustness,fixed):
    def change(pop,metric):return comparison['populations'][pop]['metrics'][metric]['comparisons']['rotation_anchor_vs_control']
    def delta(metric):return bad_initial['populations']['first8']['metrics'][metric]['delta']
    severe=change('visibility_lt_03','adds_005')
    all_add=change('all','add_01');all_strict=change('all','adds_005')
    bad=robustness['populations']['initial_bad']['metrics']['adds_005']['rotation_anchor']
    moving=fixed['populations']['fixed_parent_moving_lip_benefit']['metrics']
    values=[severe['delta'],*severe['ci95'],all_add['delta'],all_strict['delta'],
        delta('rotation_deg'),delta('center_mm'),bad['delta'],
        moving['rotation_anchor']['object_macro_percent'],moving['control']['object_macro_percent']]
    if not all(math.isfinite(x) for x in values):raise ValueError('Selection metrics must be finite')
    return dict(severe_strict_paired_ci_positive=severe['ci95'][0]>0,
        overall_add_point_not_lower=all_add['delta']>=0,
        overall_strict_point_not_lower=all_strict['delta']>=0,
        bad_initial_first8_rotation_point_not_worse=delta('rotation_deg')<=0,
        bad_initial_first8_center_point_not_worse=delta('center_mm')<=0,
        bad_initial_strict_point_not_lower=bad['delta']>=0,
        fixed_moving_point_not_lower=moving['rotation_anchor']['object_macro_percent']>=moving['control']['object_macro_percent'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--diagnostics',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    policy=json.loads(a.policy.read_text());assert policy['frozen_before_validation']
    assert sha(__file__)==policy['selection_script_sha256']
    experiment=Path(policy['experiment']);e=json.loads((experiment/'experiment.json').read_text())
    assert sha(experiment/'experiment.json')==policy['experiment_sha256']
    assert json.loads((experiment/'status.json').read_text())['phase']=='completed'
    assert json.loads((a.diagnostics/'status.json').read_text())['phase']=='completed'
    paths=dict(comparison=experiment/'comparison.json',bad_initial=a.diagnostics/'bad_initial_rotation_anchor_vs_control/analysis.json',
        robustness=a.diagnostics/'robustness/analysis.json',fixed=a.diagnostics/'fixed_groups/analysis.json')
    values={key:json.loads(path.read_text()) for key,path in paths.items()}
    assert all(value['completed'] for value in values.values())
    checks=selection_checks(**values);adopt=all(checks.values())
    name='rotation_anchor_final1000' if adopt else 'retained_startup_s1o1_seed42_final1000'
    source=experiment/'rotation_anchor/train/last.pt' if adopt else Path(policy['fallback_checkpoint'])
    expected=values['comparison']['checkpoints']['rotation_anchor'] if adopt else policy['fallback_sha256']
    assert sha(source)==expected
    a.out.mkdir(parents=True,exist_ok=False);checkpoint=a.out/'selected.pt';shutil.copy2(source,checkpoint);assert sha(checkpoint)==expected
    receipt=dict(completed=True,name=name,selected_before_test=True,test_access_before_selection=False,
        selected_at=time.time(),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=expected,source=str(source),
        policy=str(a.policy.resolve()),policy_sha256=sha(a.policy),checks=checks,new_structure_adopted=adopt,
        validation_artifacts={key:dict(path=str(path),sha256=sha(path)) for key,path in paths.items()},
        interpretation='A conservative predeclared selection rule. Non-regression point checks do not establish statistical noninferiority. No best-step or best-seed search; failed new architecture retains the previously fixed recipe. Official test cannot revise this choice.')
    (a.out/'freeze.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
