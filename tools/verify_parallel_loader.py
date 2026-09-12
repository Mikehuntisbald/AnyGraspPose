"""Compare actual augmented clips byte for byte and benchmark bounded decoding."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import torch
from lip.data.clips import ClipDataset

def digest(item):
    h=hashlib.sha256()
    for key in ('rgb','depth','poses','frames','times','k'):
        h.update(item[key].numpy().tobytes())
    h.update(json.dumps({k:item[k] for k in ('sample','seed','effective')},sort_keys=True).encode())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True)
    p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--output',required=True)
    p.add_argument('--start',type=int,default=21500);a=p.parse_args()
    torch.set_num_threads(1);cv2.setNumThreads(1)
    ds=ClipDataset(a.data_root,a.index_root,steps=64,batch=32,world=8,start=a.start,augmentation=True)
    rows=[];expected=None
    # Repeat serial after parallel so the headline does not rely on cold files.
    for threads in (1,4,1,4):
        if ds._decode_pool is not None:ds._decode_pool.shutdown();ds._decode_pool=None
        ds.decode_threads=threads;begin=time.perf_counter()
        items=ds.__getitems__(list(range(32)));elapsed=time.perf_counter()-begin
        hashes=[digest(x) for x in items]
        if expected is None:expected=hashes
        assert hashes==expected, 'Parallel loader changed pixels, sampling, augmentation, or RNG'
        row=dict(threads=threads,clips=32,seconds=elapsed,clips_per_second=32/elapsed)
        rows.append(row);print(json.dumps(row),flush=True);del items
    # Reconstruct the next resumed batch, and compare it to the uninterrupted stream.
    reference=[digest(x) for x in ds.__getitems__(list(range(32,36)))]
    ds.start+=1
    resumed=[digest(x) for x in ds.__getitems__(list(range(4)))]
    assert resumed==reference
    Path(a.output).write_text(json.dumps(dict(passed=True,bitwise_equal=True,resume_order_equal=True,
                                            clips=32,rows=rows),indent=2))

if __name__=='__main__':main()
