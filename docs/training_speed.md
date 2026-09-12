# Training throughput optimization

The speed run resumes the preserved 21,500 checkpoint in
`runs/basin_speed_v1/resume_21500.pt`. The previous runtime is retained at
`runs/basin_memory_fix/candidate/src`; the new runtime is isolated at
`runs/basin_speed_v1/candidate/src`.

The loader keeps one process per rank, prefetch factor one, and disabled
pin-memory copying. Within that worker, four threads decode and augment clips
in a batch. Each clip still owns the RNG derived from its sampler index, and
ordered thread mapping preserves batch order. This increases CPU concurrency
without multiplying the number of queued full-resolution batches or duplicating
the dataset across additional worker processes. NumPy/OpenCV/PyTorch work is
unchanged; no quantization or crop-dependent approximation is introduced.

The selected configuration keeps `preload_rollout_observations: false` and
retains the original GPU path. An additional GPU-upload reuse experiment did
not improve whole-step throughput beyond threaded decoding, so it is disabled.
Crop coordinates, camera
intrinsics, rendering, history perturbations, augmentation, loss, effective
batch 256, optimizer schedule, and the frozen critic/FP transition are unchanged.

`tools/verify_parallel_loader.py` checks 32 real augmented clips byte for byte
and checks resumed sampler order. `tools/benchmark_speed.py` runs both versions
from the same checkpoint for 30 optimizer updates on all eight GPUs. The first
eight steps are excluded from throughput statistics; per-step time uses the
slowest rank. Three rollout branches are reported separately. The comparison
also checks losses, model/optimizer tensors, scheduler, sampler position, and
container memory. Checkpoint and runtime hashes bind the accepted version.

## Measured results and validation limits

The selected threaded-only run measured 8.1099 to 4.0441 seconds per optimizer
step (2.005x) over matched steps 21,509–21,530, using the slowest of eight ranks.
Noisy-GT averaged 8.2871 to 3.5505 seconds (9 steps), LIP-only 7.2761 to 3.2763
(10 steps), and LIP→FP 10.3577 to 8.0843 (3 steps). This particular window has
only three FP steps; do not extrapolate its 2x speedup to the full remaining
schedule as FP probability rises. Peak sampled container usage was 207.94 GiB
of 512 GiB. The unchanged one-process/one-prefetch queue avoids the previous
unbounded memory configuration.

Full tests: 44 passed, 1 skipped. The generic real-FP fixture test is skipped;
actual frozen FP was exercised by the real 8-rank runs and the separate paired
GPU check. An expanded audit compared 512 clips across all eight ranks at
sampler positions 21,501 and 21,505: all RGB, depth, poses, timestamps, intrinsics,
sampling choices and augmentation seeds were byte-identical. With fixed actor
and RNG state, all three four-step rollout branches also produced identical
GPU input tensors, outputs, losses and gradients in the paired test.

The stricter multi-step trajectory comparison **did not pass**. The first
threaded-run difference was approximately 1.1e-6 in the shared gradient norm
at step 21,502, while all rank losses still matched exactly. Over 30 BF16
closed-loop updates, maximum loss difference reached 0.00368548, model tensor
difference 0.000691315, and optimizer tensor difference 0.00261232. The unchanged
baseline repeated its first eight steps exactly. The specific low-level source
of the initial gradient discrepancy has not been isolated; do not label it
proven baseline nondeterminism or claim bitwise training equivalence.

`comparison.json` and `threaded_comparison.json` preserve the failed strict
diagnostics. `approval.json` explicitly accepts functional equivalence,
unchanged training configuration, finite DDP updates, scheduler/sampler/RNG
continuity, bounded memory and throughput. It does not turn the failed
trajectory check into a pass and does not establish equal final accuracy.
Regular validation continues during production training.

The serial benchmark writes only to `runs/basin_speed_v1/baseline`. Accepted
updates from the threaded benchmark are retained at
`runs/basin_speed_v1/threaded/last.pt`; production resumes there at 21,530.
This avoids discarding the validated optimized updates. Earlier main-run log
segments remain intact; filter production rows by `run_id=basin_speed_v1`.

Evidence: `loader.json`, `input_audit.json`, `gpu_equivalence.json`,
`tests_gpu.log`, `threaded_comparison.json`, `approval.json`,
`activation.json`, `train.log`, and `memory.jsonl` under `runs/basin_speed_v1`.
These receipts, rather than this workflow description, establish what actually
passed and whether training is currently running.

Recovery after the main run has saved its next checkpoint:

```bash
cd /mnt/why/dexycb_lip
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
bash scripts/train_basin_fast.sh --resume runs/lip_v1_s0/last.pt
```

Before the first production checkpoint at 21,600, resume
`runs/basin_speed_v1/threaded/last.pt` instead. Do not launch a duplicate of an
active training job. No driver/global CUDA changes or raw-data writes are part
of this optimization.
