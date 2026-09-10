# Pure FoundationPose comparison

The comparison population is frozen by `runs/lip_v1_s0/val_10000/manifest.json` and `predictions.jsonl`: all 320 official s0 validation camera streams, 23,200 frames, first-frame GT initialization and closed-loop tracking thereafter. The test split is not used. GT first-frame rows remain in both methods' metrics, matching the existing LIP report.

`tools/eval_foundationpose.py` runs the official NVlabs FoundationPose `track_one` with two refinement iterations (the official demo default). It uses the original textured mesh, native RGB-D and camera intrinsics, official depth processing and crop generation, frozen refiner weights and official FP16 autocast. LIP uses its own fixed crop/features and BF16. Identical inputs/protocol do not imply replacing each method's native preprocessing.

Registration/scoring is not used because the first pose is provided to both trackers. A subclass skips the otherwise unused registration rotation-grid setup; the official reset and tracking functions remain unchanged. The adapter maps original-mesh poses to FoundationPose's centered mesh state. Depth changes from 1xHxW to HxW; mesh diameter is cast from NumPy scalar to Python float to avoid dtype promotion in the official crop function, without changing its numeric value. No hand annotations, GT masks, or post-initial GT poses enter tracking.

Predictions are committed before metric computation. The same `lip.evaluation.metrics.errors` uses the same cached full mesh vertices and diameter. Visibility/motion strata come from the frozen reference metric records and never enter inference. `tools/compare_foundationpose.py` rejects incomplete runs, duplicate or mismatched frames, different protocol/manifests, or different strata before comparing object macro metrics.

Environment: `.venv-fp` is separate from `.venv`. No driver or global CUDA changes. Official Drive downloads hit quota limits; the weight copy is from `gpue/foundationpose-weights`, revision `42d49e0633d245b3cf4dea6b1e7ec2b31d5b7654`, via hf-mirror.com. Download URLs and SHA-256 values are in `runs/fp_baseline_20260910/weights_receipt.json`. This establishes the exact mirror artifact used; it is not independent proof of byte identity to the quota-limited Drive files.

The background transaction `tools/run_fp_baseline_then_resume.py` waits for a fresh atomic training checkpoint, saves an immutable `resume.pt`, stops the old training workflow, evaluates on GPU 7 without concurrent training, then restarts eight-GPU training even if evaluation fails. `status.json` records the saved step and any replayed steps between checkpoint commit and interruption. Its resumed training log is `runs/fp_baseline_20260910/training_resumed.log`; evaluation and comparison logs are in that same directory. Do not start a second training job while this transaction is live.

```bash
cd /mnt/why/dexycb_lip
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src OMP_NUM_THREADS=2 .venv-fp/bin/python tools/eval_foundationpose.py \
  --fp-root third_party/FoundationPose --data-root cache/raw_full_20260910 \
  --reference runs/lip_v1_s0/val_10000 --out runs/fp_new_run
PYTHONPATH=src .venv-fp/bin/python tools/compare_foundationpose.py \
  --lip runs/lip_v1_s0/val_10000 --fp runs/fp_new_run --out runs/fp_new_run/comparison.json
```

Interpretation: LIP has been trained on DexYCB s0 train, while FoundationPose is evaluated with frozen released weights. Report validation performance as such, not as a final held-out test or equal-training-data experiment. Latency excludes object setup and metric computation; FoundationPose includes its depth filtering and refinement, while LIP includes its feature construction and rendering. Keep the original method-specific latency scopes with the numbers.
