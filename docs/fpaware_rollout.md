# FP-aware continuation from checkpoint 19,000

User-authorized schedule change: all three branches train four updates. At step 19,000 the optimizer-step mixture is 40% noisy GT history, 30% LIP-only history, 30% LIP-to-FP history. It linearly approaches 20%, 20%, 60% at step 40,000. Selection is deterministic from the shared seed plus optimizer step, so all DDP ranks take the same branch. This is a mixture across optimizer steps, not an exact per-batch sample quota.

Every update computes the original pose loss on the current LIP prediction against current GT. In the FP branch only, after backward, detached LIP output is refined by the frozen official FoundationPose refiner using the current RGB-D observation. The detached refined state enters the next history. No gradient traverses FP, no FP parameters enter the optimizer, and no GT after initial history enters either recursive branch's state transition. The last update needs no FP call because no further history is consumed. The noisy-GT branch refreshes strict-past noisy GT at each update; both recursive branches bootstrap with the same type of strict-past noisy history.

FoundationPose remains outside Tracker. Its refiner is in eval mode with requires_grad=False, official tracking uses two iterations and FP16 AMP, and RNG/default tensor type changes are isolated from LIP. Checkpoints save LIP/optimizer/scheduler/RNG state and the external FP weight SHA-256 in config, not a trainable FP state. Per-object FP render caches are process-local and reused; every transition resets the estimator state to that sample's detached LIP prediction, so clips cannot share pose history.

Only the start checkpoint is reused from the prior experiment. Six-step DDP preflight outputs are separate from the formal continuation. Earlier model/checkpoint/config receipts are retained. Full validation in lip.train remains the standalone LIP closed-loop validation, so it remains comparable with earlier reports; it is not an LIP+FP inference evaluation.

Runtime locations:
- Candidate source: runs/fpaware_19000/candidate/src
- Config: runs/fpaware_19000/candidate/configs/fpaware_19000.yaml
- Preflight: runs/fpaware_19000/{all_tests.log,probe.json,ddp_probe.log,preflight_passed.json}
- Formal log: runs/fpaware_19000/train.log
- Checkpoints and rank logs: runs/lip_v1_s0/

After interruption (do not duplicate a live job):
```bash
cd /mnt/why/dexycb_lip
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
bash scripts/train_fpaware_8gpu.sh --resume runs/lip_v1_s0/last.pt
```

First resumed run uses the immutable runs/fpaware_19000/resume.pt at step 19,000. Later recovery uses last.pt after a new checkpoint has been saved.
