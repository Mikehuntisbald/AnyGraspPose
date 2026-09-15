"""Freeze train fragments and their exact initial-image prediction requests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips
from lip.data.external_initializers import initializer_key,request_real_initializer
from lip.engine.config import check_data_gate


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('data-root','index-root','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--count',type=int,default=64000);p.add_argument('--counter-start',type=int,default=512000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--real-probability',type=float,default=.5);a=p.parse_args()
    audit=check_data_gate(a.index_root);data=StreamClips(a.data_root,a.index_root,burn_in=8,unroll=48,seed=a.seed,decode_threads=0)
    samples=[data.choose(i) for i in range(a.counter_start,a.counter_start+a.count)];requests={};requested=0
    for item in samples:
        if not request_real_initializer(item['seed'],a.real_probability):continue
        stream=data.streams[item['stream']];frame=int(data.poses[item['stream']]['frames'][item['start']]);key=initializer_key(stream['stream_id'],frame)
        requests[key]=dict(key=key,stream_id=stream['stream_id'],frame_index=frame,object_id=stream['object_id']);requested+=1
    a.out.mkdir(parents=True,exist_ok=False);manifest=a.out/'training_samples.json';manifest.write_text(json.dumps(samples))
    report=dict(completed=True,split='train',split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        training_manifest=str(manifest.resolve()),training_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        counter_start=a.counter_start,draws=a.count,seed=a.seed,real_probability=a.real_probability,
        requested_draws=requested,unique_requests=len(requests),requests=[requests[k] for k in sorted(requests)],
        scope='Exact train fragment initial images; no val/test images or poses. Real requests chosen before detector outputs or GT-error filtering.')
    (a.out/'requests.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='requests'}))


if __name__=='__main__':main()
