import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('natural_visibility',Path(__file__).resolve().parents[1]/'tools/collect_natural_visibility.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_legacy_endpoint_uses_frame_ids_and_unknown_is_distinct_from_missing():
    frames=[np.arange(4,40)]
    clips=[dict(stream=0,start=9,stride=1,visibility=None),dict(stream=0,start=2,stride=2,visibility=None)]
    values,known=module.unpack_legacy([{}],frames,clips)
    assert known[0][12] and values[0][12] is None and not known[0][11]
    clips.append(dict(stream=0,start=9,stride=1,visibility=.3))
    with pytest.raises(ValueError,match='Contradictory'):module.unpack_legacy([{}],frames,clips)


def test_box_containment_is_conservative_and_rejects_behind_camera():
    mesh=dict(vertices=np.array([[-.1,-.1,-.1],[.1,.1,.1]]));pose=np.eye(4)[None];pose[:,2,3]=1
    k=np.array([[100.,0,320.],[0,100.,240.],[0,0,1.]])
    assert module.bbox_inside(mesh,pose,k)==[True]
    pose[:,0,3]=10;assert module.bbox_inside(mesh,pose,k)==[False]
    pose[:,2,3]=-.1;assert module.bbox_inside(mesh,pose,k)==[False]
