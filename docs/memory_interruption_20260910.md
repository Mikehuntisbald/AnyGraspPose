# 2026-09-10 training interruption

Kernel evidence, captured in `runs/basin_memory_fix/kernel_before.log`, establishes a container memory OOM at 19:07:34 CST. Usage and limit were both 536870912 KiB (512 GiB). The killed host PIDs were 1038221 and 1038224. Rank 2 and rank 5 subsequently reported SIGKILL. NVSwitch PRIV errors, Xid 137/94 and NCCL peer-memory errors followed at 19:07:43. This timeline supports OOM as the initiating failure; it does not establish a defective GPU. The earlier 17:59:50 interruption also has an explicit container OOM record. A zero failcnt and ample host-wide free RAM do not rule out container OOM.

At failure, cgroup shmem was 334410534912 bytes (311.4 GiB), anonymous memory 197231353856 bytes (183.7 GiB). The loader queued full-resolution float RGB-D, 11 frames per clip, batch 32, four workers and prefetch factor two per rank. Across eight ranks this permits 64 prefetched batches, before current batches, pinned copies, worker processes and other allocations. `train.py` ignored configured pin-memory and prefetch settings by hardcoding them. Fifteen orphaned multiprocessing workers/resource trackers from the two stopped training runs were identified by PPid=1, project interpreter and project cwd, then terminated; other processes were not targeted.

The fix makes the loader honor config settings. The recovery config uses one worker per rank, prefetch factor one and no pinned-memory copies. Batch size 32 per GPU, eight ranks, sampling, optimizer schedule, frozen critic, basin weight and FP-aware rollout remain unchanged. Checkpoint interval is 100 steps. No NCCL transport workaround, global CUDA/driver change, GPU reset or raw-data modification was applied.

Runtime: `runs/basin_memory_fix/candidate/src`; config: `runs/basin_memory_fix/config.yaml`. The immutable checkpoint `resume_21000.pt` is the last valid pre-crash state. Thirty real training steps in the isolated probe exercise all three rollout branches; the gated supervisor only starts main training after all eight ranks and tests pass, then resumes the probe's saved 21030 checkpoint. Continuous container RSS/shmem/usage measurements go to `memory.jsonl`; `preflight.json`, `tests.log`, `probe.log`, `train.log` and `status.json` are the evidence. Long-run stability must be assessed from continuing measurements, not inferred from a short probe.

Recovery after a future stop (do not launch a duplicate while training is alive):

```bash
cd /mnt/why/dexycb_lip
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
bash scripts/train_basin_memory_fixed.sh --resume runs/lip_v1_s0/last.pt
```

Before the first new main checkpoint, use `runs/basin_memory_fix/probe/last.pt` to retain the validated 30 probe updates. Original interruption logs and runtime snapshots remain preserved. Current main log rows are identified by `run_id=basin_memory_fix`; overlapping step numbers in older segments are not current progress.
