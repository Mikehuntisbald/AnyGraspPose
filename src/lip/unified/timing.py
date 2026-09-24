"""CUDA event pairs collected without synchronizing every module."""
from contextlib import contextmanager
import torch


class Timings:
    def __init__(self):self.events=[]
    @contextmanager
    def record(self,name):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        try:yield
        finally:b.record();self.events.append((name,a,b))
    def seconds(self):
        result={}
        for name,a,b in self.events:result[name]=result.get(name,0.)+a.elapsed_time(b)/1000
        return result
