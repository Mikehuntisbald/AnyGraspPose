"""Give each torchrun worker one visible GPU before importing torch.

All compiler tensor devices then have logical index zero, allowing one cache
identity across workers. Global RANK/WORLD_SIZE and data sharding are unchanged.
"""
import os
import runpy
import sys


def main():
    local=int(os.environ.get('LOCAL_RANK',0))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',') if visible is not None else None
    if devices is not None and local>=len(devices):raise ValueError('Insufficient visible GPUs for local rank')
    os.environ['CUDA_VISIBLE_DEVICES']=devices[local] if devices is not None else str(local)
    os.environ['LOCAL_RANK']='0'
    if len(sys.argv)<2:raise ValueError('Expected worker entrypoint')
    sys.argv=sys.argv[1:]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
