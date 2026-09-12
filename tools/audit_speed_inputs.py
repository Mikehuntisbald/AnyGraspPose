"""Audit both first-divergence batches on every DDP rank, outside GPU training."""
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import torch
from lip.data.clips import ClipDataset

def hashes(item):
    return {k:hashlib.sha256(item[k].numpy().tobytes()).hexdigest() for k in ('rgb','depth','poses','frames','times','k')} | {
        k:json.dumps(item[k],sort_keys=True) for k in ('sample','seed','effective')}

def audit(rank):
    torch.set_num_threads(1)
    ds=ClipDataset('cache/raw_full_20260910','cache/dexycb_s0',steps=32,batch=32,world=8,rank=rank,start=21501)
    rows=[]
    for position in (21501,21505):
        ds.start=position;ds.decode_threads=1
        items=ds.__getitems__(list(range(32)));reference=[hashes(x) for x in items];del items
        ds.decode_threads=4;items=ds.__getitems__(list(range(32)));changed=[]
        for i,item in enumerate(items):
            result=hashes(item)
            fields=[k for k in reference[i] if reference[i][k]!=result[k]]
            if fields:
                serial=ds[i];details={k:dict(max_abs=(serial[k]-item[k]).abs().max().item(),
                                          unequal=int((serial[k]!=item[k]).sum()))
                                     for k in fields if isinstance(serial[k],torch.Tensor)}
                changed.append(dict(index=i,fields=fields,details=details))
        row=dict(rank=rank,position=position,changed=changed,clips=32);rows.append(row)
        del items
    Path(f'runs/basin_speed_v1/input_audit_rank{rank}.json').write_text(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[r for result in pool.map(audit,range(8)) for r in result]
    receipt=dict(passed=all(not r['changed'] for r in rows),clips=512,rows=rows)
    Path('runs/basin_speed_v1/input_audit.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
