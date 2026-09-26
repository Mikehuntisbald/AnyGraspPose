from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,numpy as np
root=Path('/mnt/why/dexycb_lip/cache/dexycb_s0')
rows=[r for r in map(json.loads,(root/'streams.jsonl').read_text().splitlines()) if r['split']=='train']
def check(r):
 with np.load(root/r['pose_cache'],allow_pickle=False) as z:f=z['frames']
 mismatch=f!=np.arange(len(f))
 return dict(stream=r['stream_id'],frames=len(f),mismatched_positions=int(mismatch.sum()),maximum_offset=int(np.max(np.abs(f-np.arange(len(f))))))
with ThreadPoolExecutor(8) as p:out=list(p.map(check,rows))
bad=[r for r in out if r['mismatched_positions']]
result=dict(streams=len(out),bad_streams=len(bad),mismatched_positions=sum(x['mismatched_positions'] for x in out),examples=bad[:20],scope='Training partition pose-cache frame indices versus implicit native array indexing')
path=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/official_flow_v65/experiment/frame_index_audit.json')
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
