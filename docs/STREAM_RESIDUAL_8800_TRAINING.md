# Function-preserving cross residual from dual 8,800

The independent architecture `stream_dual_cross_residual` keeps the trained dual readout:

```
base = O + sigmoid(old_gate([O, C])) * old_context_projection(C)
H = CrossAttention(O, context_history)
z = base + sigmoid(new_gate([O, H])) * H
```

The old gate, context projection, queries, encoders, temporal blocks and pose head load unchanged from the archived dual 8,800 checkpoint. Only the new cross-attention parameters and `cross_gate` are initialized. The cross output projection is zero initially and the new gate is nonzero (sigmoid(-2)), preserving the complete parent function while allowing the output projection to receive gradients. Source KV and context KV remain separate; this architecture has its own cache contract.

The first formal experiment runs 1,000 steps with all parent learning rates zero. Only the added branch changes. Zero learning rates also disable AdamW decay on the parent; gradients and optimizer moments are still computed. Batch=8/GPU on eight H20s, effective 64 sequences, 8 burn-in + 16 supervised frames, BF16. This phase uses the existing seed and sampler policy. The source implementation allows later joint training, but this experiment stops at 1,000 steps and evaluates first.

Before formal training, `tools/verify_residual_start.py` requires identical saved parent/initial predictions for all 320 streams and 23,200 frames, identical object-macro metrics, exact checkpoint hashes and byte-identical parent tensors. A completed architecture-specific preflight is required separately. After the 1,000-step training stage, `tools/train_residual_stage.py` checks every parent tensor again, launches eight-shard full s0 validation, and writes a paired-population comparison. No FP, critic, current-frame GT predictor input, or depth correction is introduced.

Runtime: `/mnt/why/dexycb_lip/stream_v2_residual_dual_8800`. Data/cache and Python environment are symlinked to existing project resources. Older training/evaluation runtimes are retained.

```bash
cd /mnt/why/dexycb_lip/stream_v2_residual_dual_8800
.venv-fp/bin/python tools/train_residual_stage.py \
  --config configs/stream_lip_v2_residual_8800.yaml \
  --init runs/stream_v2_residual_8800_init/init.pt \
  --parent /mnt/why/dexycb_lip/stream_v2/runs/stream_v2_dual_archive/checkpoint_8800.pt \
  --baseline /mnt/why/dexycb_lip/stream_v2/runs/stream_v2_dual_8800_s0_val_review \
  --equivalence runs/stream_v2_residual_init_s0_val/parent_equivalence.json \
  --data-root /mnt/why/dexycb_lip/cache/raw_full_20260910 \
  --index-root cache/dexycb_s0 \
  --out runs/residual_8800_frozen_1000 --steps 1000
```

This command creates a new experiment directory and refuses to overwrite an existing one. Inspect `status.json` before launching; do not duplicate an active run. Training metrics/checkpoints live in `train/`, full evaluation in `s0_val/`, and the final comparison/report in the experiment root. A successful comparison is evidence for this one frozen-parent run, not a guarantee of general architectural superiority.

## Verified execution

- Full initialization validation: 320 streams / 23,200 frames; every pose exactly matches the archived dual 8,800 parent (`max_abs_pose_difference=0`). ADD@0.1d=80.8108147%, ADD-S@0.1d=96.5688562%, center=9.9355856 mm, rotation=12.6420484 degrees (object macro excluding initialization).
- 117 distinct tests passed: 109 CPU and 8 additional CUDA cases, including exact parent equivalence under FP32 and BF16. Three CPU cases were repeated in the CUDA command and are not double-counted.
- Real 16-fragment / 300-step preflight: loss 0.0361253 -> 0.0325664, center 2.3155 -> 2.1091 mm. These are training-fragment diagnostics, not validation gains.
- Eight-GPU 50-step preflight and three-step resume passed; all eight ranks restored RNG/optimizer. All 407 parent state tensors remained bitwise unchanged at step 53.
- Formal training was launched in `runs/residual_8800_frozen_1000`; its live `status.json` is authoritative. Post-training validation is automatic but is not claimed complete here.

Local evidence: [preflight summary](../runs/stream_v2_residual_8800_preflight/summary.json), [full-data equivalence](../runs/stream_v2_residual_init_s0_val/parent_equivalence.json), and [launch command](../runs/residual_8800_launch.json).

To resume an interrupted training phase to the same 1,000-step boundary (only after verifying the old process has exited):

```bash
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_residual_8800.yaml \
  --resume runs/residual_8800_frozen_1000/train/last.pt \
  --output runs/residual_8800_frozen_1000/train --max-steps 1000
```

This direct resume command restores training only; the failed supervisor does not automatically resume. Check its status before running the evaluation entry point afterward. A normally running supervisor already schedules evaluation and should not be duplicated.

The first formal checkpoint at **step 100** was archived and checked: all 407 parent tensors remain unchanged. At the following snapshot all eight ranks had reached step 150, approximately 1.08 s/step; the new residual norm was about 0.085–0.089. See [checkpoint verification and live-run snapshot](../runs/residual_8800_frozen_1000/first_checkpoint_verified.json). This snapshot is not a post-training accuracy result.
