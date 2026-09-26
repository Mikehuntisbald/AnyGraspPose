"""Independent native OBJ/annotation projection; no crop or neural renderer."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image
ROOT=Path('/mnt/why/dexycb_lip/cache/raw_full_20260910')
INDEX=Path('/mnt/why/dexycb_lip/cache/dexycb_s0')
OUT=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/official_flow_v65/experiment')
sid='20200903-subject-04/20200903_105022/836212060125';frame=48
stream=next(s for s in map(json.loads,(INDEX/'streams.jsonl').read_text().splitlines()) if s['stream_id']==sid)
audit=json.loads((INDEX/'audit.json').read_text())
vertices=[]
with (ROOT/stream['mesh_path']).open() as f:
 for line in f:
  if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
vertices=np.array(vertices)*audit['mesh_scale_to_m']
with np.load(ROOT/sid/f'labels_{frame:06d}.npz',allow_pickle=False) as z:pose=z['pose_y'][stream['object_index_in_sequence']].astype(float)
pose[:,3]*=audit['pose_scale_to_m']
xyz=vertices@pose[:,:3].T+pose[:,3]
k=np.array(stream['intrinsics']);uvh=xyz@k.T;uv=uvh[:,:2]/uvh[:,2:]
assert (xyz[:,2]>0).all() and np.isfinite(uv).all()
image=cv2.cvtColor(cv2.imread(str(ROOT/sid/f'color_{frame:06d}.jpg')),cv2.COLOR_BGR2RGB)
hull=cv2.convexHull(np.round(uv).astype(np.int32))
cv2.polylines(image,[hull],True,(255,0,255),2)
Image.fromarray(image).save(OUT/'native_annotation_projection.png')
(OUT/'native_projection_receipt.json').write_text(json.dumps(dict(stream=sid,frame=frame,object_id=stream['object_id'],vertices=len(vertices),
 native_obj_and_annotation=True,crop_transform_used=False,neural_renderer_used=False,pose_or_label_changed=False,
 scope='Convex hull of native mesh vertices over raw RGB; this does not independently certify annotation accuracy.'),indent=2)+'\n')
