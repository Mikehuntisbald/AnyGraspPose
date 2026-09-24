"""One-batch CPU prefetch of exact native frames, never pose-conditioned features."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
import torch
from lip.data.jepa_rigid_episodes import RigidEpisodeFactory
from lip.engine.jepa_cuda_replay import CUDA_CAPTURE_LOCK


class FastRigidEpisodeFactory(RigidEpisodeFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool.shutdown()
        self.pool = ThreadPoolExecutor(8)
        self.prefetch_pool = ThreadPoolExecutor(2)
        self.pending = {}
        self.transfers = []
        self.cuda_device = torch.cuda.current_device()
        if self.packed is not None:
            stream = self.packed.stream
            lock = Lock()
            def locked_stream(sid):
                with lock:
                    return stream(sid)
            self.packed.stream = locked_stream

    def prefetch_rigid(self, seeds, supervised=8, burn=32):
        for seed in seeds:
            key = (seed, supervised, burn)
            if key not in self.pending:
                if len(self.pending) >= 4:
                    raise ValueError('Rigid prefetch is bounded to one four-lane microbatch')
                self.pending[key] = self.prefetch_pool.submit(self.prepare_pinned_cpu, seed, supervised, burn)

    def prepare_pinned_cpu(self, seed, supervised, burn):
        *metadata, arrays = super().prepare_cpu(seed, supervised, burn)
        with CUDA_CAPTURE_LOCK, torch.cuda.device(self.cuda_device):
            pinned = tuple(torch.from_numpy(np.ascontiguousarray(x)).pin_memory() for x in arrays)
        return (*metadata, pinned)

    def prepare_cpu(self, seed, supervised=8, burn=32):
        future = self.pending.pop((seed, supervised, burn), None)
        return future.result() if future is not None else self.prepare_pinned_cpu(seed, supervised, burn)

    def upload(self, array):
        host = array
        value = host.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.transfers = [(event, host) for event, host in self.transfers if not event.query()]
        self.transfers.append((event, host))
        return value
