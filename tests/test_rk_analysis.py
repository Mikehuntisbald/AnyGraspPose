import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'tools/compare_rk_ablation.py'
spec=importlib.util.spec_from_file_location('rk_analysis',path)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def test_factorial_contrasts_and_paired_object_macro():
    values=np.array([[10,12,13,20],[10,12,13,20],[30,32,33,40]],dtype=float)
    objects=np.array([1,1,2]);sequences=np.array(['a','b','c'])
    point,draws=analysis.paired_summary(values,objects,sequences,np.array([[1,1,1],[2,0,1],[0,0,3]]))
    np.testing.assert_allclose(point,[20,22,23,30])
    np.testing.assert_allclose(draws@analysis.CONTRASTS['interaction'],[5,5,5])
    assert point@analysis.CONTRASTS['R_at_K0']==2
    assert point@analysis.CONTRASTS['R_at_K1']==7


def test_long_occlusion_recovery_windows_never_cross_stream_or_missing_visibility():
    def row(i,s='a',v=.2):return dict(stream_id=s,frame_index=i,visibility=v)
    rows=[row(i) for i in range(9)]+[row(9,v=.8),row(10,v=None)]
    rows += [row(i,'b',.8) for i in range(20)]
    keys,episodes=analysis.episode_populations(rows)
    assert len(keys)==9 and len(episodes)==1
    assert episodes[0]['post_keys']==[('a',9)] and episodes[0]['right_censored']
    assert analysis.episode_populations([row(i) for i in range(8)])[1]==[]
