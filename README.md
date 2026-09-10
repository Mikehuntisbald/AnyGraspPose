# AnyGraspPose — DexYCB-LIP v1

An implemented deterministic RGB-D temporal object-pose updater. This is a new engineering model, not a claim of reproducing a published method. See [the frozen task](docs/TASK.md), [the matched FoundationPose comparison](runs/fp_baseline_20260910/REPORT.md), and [the baseline protocol](docs/foundationpose_baseline.md). Historical subset reports are retained as historical diagnostics.

The fixed network is ResNet50 through layer3 (ImageNet V2), a 9-channel geometry CNN, two spatial RGB-query/geometry-key-value cross-attention blocks, four causal temporal Transformer layers, a 256-dimensional readout and a zero-initialized 6-output correction head. Geometry is rendered once per window at the latest accepted estimate. RGB remains unmasked. There are no MANO imports, hand inputs or hand losses in `src/lip`.

The raw dataset is read-only. Output, metadata, pretrained weights, renderer build files and temporary files live under this project. No driver, global CUDA or global PyTorch changes are required. The remote `.venv` uses the existing PyTorch installation read-only and installs extra packages locally. Use the environment snapshot in `runs/environment.json` and `docs/requirements-remote.txt`; the latter records the actual environment rather than a portable CUDA installer.

```bash
cd /mnt/why/dexycb_lip
source scripts/env.sh
export DEX_YCB_DIR=/path/to/extracted/DexYCB
```

Repository: https://github.com/Mikehuntisbald/AnyGraspPose. The training server uses a separately synchronized runtime copy; publishing this repository does not change its running processes or Git state.


## Verified execution and validation

Full official s0 preflight passed: 10 subjects, 1,000 physical sequences; 32-sample geometry checks; 28 unit tests passed and one optional test skipped; real 32-clip / 500-step overfit; GPU memory probes; eight-GPU 50-step smoke run and resume to step 53. Three additional baseline comparison checks passed. The resolved configuration uses eight H20 GPUs, batch 32 per GPU, effective batch 256 and a 40,000-step target. Training is ongoing; this snapshot does not claim completion.

Matched full validation (320 camera streams, 23,200 frames), GT first-frame initialization followed by closed-loop tracking, object macro averages:

| Metric | LIP at step 10,000 | Frozen FoundationPose |
|---|---:|---:|
| ADD-S < 0.1 diameter | 95.42% | 85.98% |
| ADD < 0.1 diameter | 76.83% | 69.05% |
| Center error | 11.33 mm | 40.56 mm |
| Rotation error | 13.80 degrees | 20.01 degrees |
| Lost-frame rate | 3.36% | 11.81% |

LIP is trained on DexYCB s0 train; FoundationPose uses frozen released weights from a pinned mirror. These are validation results, not a final test or equal-training-data experiment. Exact metrics, weight provenance and logs are under `runs/fp_baseline_20260910/`. Dataset archives, model checkpoints, installed dependencies and generated images are not versioned. Install optional FoundationPose from the official NVlabs repository and follow the pinned provenance in the baseline documentation.

For a fresh compatible environment:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# Optional GPU backend, install the pinned checkout listed in docs/upstream/versions.json:
python -m pip install --no-build-isolation ./cache/nvdiffrast
```

Full data preflight, in the foreground:

```bash
bash scripts/preflight.sh
```

This runs inventory audit, official indexing, 32-example geometric audit, sampling statistics, unit tests, real 32-clip/500-step overfit, real single-GPU batch probes, an 8-GPU/50-step U=4 run, and a resume to step 53. It then approves the resolved config. It does **not** start the 40,000-step training. The test split is not used by this preflight.

The corresponding individual commands are:

```bash
python tools/audit_data.py --data-root "$DEX_YCB_DIR" --out runs/audit
python tools/build_index.py --data-root "$DEX_YCB_DIR" --setup s0 --out cache/dexycb_s0
python tools/check_geometry.py --data-root "$DEX_YCB_DIR" --index cache/dexycb_s0 --num-samples 32 --out runs/geometry
python tools/build_sampling.py --data-root "$DEX_YCB_DIR" --index cache/dexycb_s0 --length 8
python -m pytest -q
python tools/overfit_small.py --config configs/dexycb_lip_v1.yaml --num-clips 32 --steps 500
python tools/probe_batch.py --config configs/dexycb_lip_v1.yaml --out configs/resolved_8gpu.yaml
bash scripts/smoke_8gpu.sh
python tools/verify_preflight.py --config configs/resolved_8gpu.yaml --approve
```

The formal training and resume commands, after full-data preflight has passed, are:

```bash
bash scripts/train_8gpu.sh
bash scripts/train_8gpu.sh --resume runs/lip_v1_s0/last.pt
```

These refuse to run when real full-s0 preflight is absent or stale. `python -m torch.distributed.run` is the torchrun implementation, invoked with the selected virtual-environment Python to avoid a global console-script shebang bypassing that environment. Checkpoints contain model, optimizer, scheduler, global step, config, sampling position, all-rank RNG state, mesh/split hashes, code hash and RGB-weight identity. `last.pt` is atomic. `best.pt` is selected only by **full validation macro-over-object ADD-S recall at 0.1 diameter**. ADD and rotation remain in the same report. Quick validation never selects best.

After a trained checkpoint exists:

```bash
python -m lip.evaluate --config configs/resolved_8gpu.yaml --checkpoint runs/lip_v1_s0/best.pt --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 --split val --mode one-step --out runs/eval_val_onestep
python -m lip.evaluate --config configs/resolved_8gpu.yaml --checkpoint runs/lip_v1_s0/best.pt --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 --split val --mode closed-loop --out runs/eval_val_closed
# Only after model/hyperparameters are finalized:
python -m lip.evaluate --config configs/resolved_8gpu.yaml --checkpoint runs/lip_v1_s0/best.pt --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 --split test --test-finalized --mode closed-loop --out runs/eval_test_closed
```

For final test one-step evaluation, explicitly create its fixed sampling population with `python tools/build_sampling.py --data-root "$DEX_YCB_DIR" --index cache/dexycb_s0 --length 8 --test-finalized`, then use `--mode one-step --test-finalized`. Full-sequence evaluation needs no clip population. Every valid GT frame remains in `predictions.jsonl`, including large-error/lost frames. The first valid GT initializes the tracker once; all subsequent accepted states come from the updater. Nonfinite outputs retain the last accepted pose. Segmentation and GT rendering are accessed in the separate metric branch. Overlays are generated from the same saved predictions. Latency is synchronized batch=1 wall time, excluding metrics/visualization and including raw frame decode.

One-step evaluation compares learned, zero-motion and actual-time constant-velocity predictions with identical seeded noisy histories. Full-sequence reporting includes center error (mm), rotation (degrees), ADD/ADD-S (meters), recalls at 0.05d/0.1d, object/stream/physical-sequence summaries, known/unknown visibility, moving/nonmoving strata and first five-consecutive-frame ADD-S >0.1d failure. This failure threshold is a benchmark convention, not an industrial tolerance. Rotation supervision is canonical GT, without guessed symmetries.

Single-frame and reduced-context ablations are `configs/dexycb_lip_v1_single_frame.yaml` and `configs/dexycb_lip_v1_reduced_context.yaml`. Generate the matching `--length 1` sampling cache for the former and run separate preflight/training; no ablation long runs are launched automatically.

Data contracts: grasp target is `meta.ycb_ids[meta.ycb_grasp_ind]`; `pose_y[meta.ycb_grasp_ind]` is used independently of class ID. Official s0 sequence ordering requires 100 sequences per subject. Incomplete listings are never silently reindexed. Depth conversion defaults to 0.001 and is separately checked against rendered visible-object depth; mesh and pose each have their own scale. Geometry uses OpenCV axes and meters. The mesh center is the axis-aligned bbox midpoint; diameter is the farthest distance between convex-hull vertices, not the bbox diagonal. Loss points are 512 fixed area-weighted surface samples; evaluation uses all mesh vertices.

A bounded **verified train subset** diagnostic is supported only for the four training sequences explicitly listed at sorted positions 0..3 in the pinned official README. `--verified-train-subset` indexes that known subset; tools require explicit `--allow-verified-subset`. Training additionally requires `--max-steps <=100`. Such runs are never allowed to approve the full-s0 config. Their logs and checkpoints carry subset provenance. See `runs/STATUS.md` for actual execution and exact commands.

FoundationPose is optional and independent. `FoundationPoseAdapter` validates the actual `track_one(rgb, depth, K, iteration)` signature, converts original to FP-centered poses and restores internal state after rejected/failed refinement. Nonzero-center unit tests do not imply real FP inference has run. The upstream snapshot retains its license in `docs/upstream/`; no upstream hand implementation is imported by the main model.

Motion speed cutoffs for evaluation are frozen at the predeclared 75th percentile of adjacent **training** transitions (center speed in diameters/second and angular speed in radians/second). They are stored with the split hash in `motion_thresholds.json`; val/test never set these cutoffs. Moving-clip sampling keeps the task's separate 0.05d / 5-degree window thresholds. `clips_per_sec` in rank logs is per rank; `observed_frames_per_sec` counts repeated window-frame visits during rollout, not unique dataset frames.

The 2026-09-10 continuation uses completed uploads for subjects 02, 03, and 10. Their 100-sequence inventories retain original global subject indices; s0 membership is train 240 / val 20 / test 40 physical sequences. The full 10-subject gate remains closed. The current `resolved_8gpu.yaml` records this exact scope and the latest real-data receipts. See `runs/available_20260910/STATUS.md` for metrics and commands.

To index such a declared subject subset, use `tools/build_index.py --available-subjects ... --workers 12`; each selected subject must contain all 100 official sequences. This differs from the legacy four-known-sequence diagnostic. `build_sampling.py` can be launched with torchrun on 8 GPUs; run `tools/merge_sampling.py --world 8` only after every shard finishes. Shard ownership, duplicate clips, and split hashes are checked. Test sampling requires the existing explicit finalization flag.

Quick validation subsets are now seeded and object-balanced rather than a directory prefix. Each evaluation manifest records the checkpoint SHA256/global step and config. Full validation still determines best checkpoints; best selection state is atomically retained in both last and best so resume cannot forget the previous best score.
