"""Bounded CUDA replay of deterministic, no-grad subgraphs only.

Trainable blocks use replay during burn-in; their supervised forward/backward
remain eager. Autocast casts execute inside every replay so parameter updates
are observed. Returned tensors own storage, including frozen DINO features.
"""
import torch
from threading import RLock

# Pinned host allocation also calls CUDA. It must not overlap global capture.
CUDA_CAPTURE_LOCK = RLock()


class NoGradReplay:
    def __init__(self, call, force_no_grad=False, require_full_memory=False):
        self.call = call
        self.force_no_grad = force_no_grad
        self.require_full_memory = require_full_memory
        self.cache = {}

    def __call__(self, *args):
        if (torch.is_grad_enabled() and not self.force_no_grad) or args[0].device.type != 'cuda':
            return self.call(*args)
        if self.require_full_memory and (args[4] is None or args[4].shape[1] != 128):
            return self.call(*args)
        enabled = torch.is_autocast_enabled('cuda')
        dtype = torch.get_autocast_dtype('cuda')
        key = (enabled, dtype, tuple(None if x is None else (tuple(x.shape), x.dtype, x.device) for x in args))
        with torch.no_grad(), torch.autocast('cuda', enabled=enabled, dtype=dtype, cache_enabled=False):
            if key not in self.cache:
                # Bound graph memory for unusual diagnostic shapes. Eager is exact.
                if len(self.cache) >= 2:
                    return self.call(*args)
                static = tuple(None if x is None else x.detach().clone() for x in args)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.call(*static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with CUDA_CAPTURE_LOCK, torch.cuda.graph(graph):
                    outputs = self.call(*static)
                self.cache[key] = (static, graph, outputs)
            static, graph, outputs = self.cache[key]
            for target, source in zip(static, args):
                if target is not None:
                    target.copy_(source)
            graph.replay()
            if isinstance(outputs, tuple):
                return tuple(x.clone() for x in outputs)
            return outputs.clone()


def install_rigid_replay(core):
    core.encoder.forward = NoGradReplay(core.encoder.forward, force_no_grad=True)
    for block in core.blocks:
        block.forward = NoGradReplay(block.forward, require_full_memory=True)
