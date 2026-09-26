from types import SimpleNamespace
import io
import numpy as np
import torch,cv2
from lip.geometry.so3 import exp
from lip.unified.full_window import choose_start,transport_centered_error,decode_frame
from lip.data.jepa_rigid_cache import RigidFrameCache


def test_full_window_covers_all_legal_starts_reproducibly():
    values=[choose_start(s,72,10) for s in range(4096)]
    assert set(values)==set(range(63)) and values==[choose_start(s,72,10) for s in range(4096)]
    assert {start+offset for start in values for offset in (0,1,8,9)}==set(range(72))
    assert choose_start(42,10,10)==0
    try:choose_start(42,9,10)
    except ValueError:pass
    else:raise AssertionError('Accepted a crossing window')


def test_center_error_transport_preserves_independent_rotation_and_translation():
    a=torch.eye(4);a[:3,:3]=exp(torch.tensor([.2,-.1,.3]));a[:3,3]=torch.tensor([.3,.1,.8])
    b=torch.eye(4);b[:3,:3]=exp(torch.tensor([-.4,.2,.1]));b[:3,3]=torch.tensor([-.2,.2,1.2])
    estimate=a.clone();err=exp(torch.tensor([.04,-.02,.1]));shift=torch.tensor([.01,-.02,.03])
    estimate[:3,:3]=err@a[:3,:3];estimate[:3,3]+=shift
    result=transport_centered_error(estimate,a,b)
    torch.testing.assert_close(result[:3,:3]@b[:3,:3].T,err)
    torch.testing.assert_close(result[:3,3]-b[:3,3],shift)


def test_uncached_late_frame_uses_exact_native_bytes(tmp_path):
    rgb=np.arange(18*20*3,dtype=np.uint8).reshape(18,20,3);depth=np.arange(18*20,dtype=np.uint16).reshape(18,20)
    seg=np.zeros((18,20),dtype=np.uint8);seg[:,5:]=4
    cv2.imwrite(str(tmp_path/'color_000065.jpg'),rgb);cv2.imwrite(str(tmp_path/'aligned_depth_to_color_000065.png'),depth)
    np.savez(tmp_path/'labels_000065.npz',seg=seg)
    factory=SimpleNamespace(root=tmp_path,packed=RigidFrameCache)
    stream=dict(relative_dir='.',object_id=4)
    direct,cached=decode_frame(factory,'stream',stream,65,{})
    assert cached is False
    entries={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    packed,cached=decode_frame(factory,'stream',stream,65,entries)
    assert cached is True
    assert all(np.array_equal(x,y) for x,y in zip(direct,packed))
