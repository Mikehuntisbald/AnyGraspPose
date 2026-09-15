# Full native s0 val with real non-GT initialization

This protocol screens frozen LIP candidates on all **320 grasped-object camera
streams / 23,200 frames**. It uses actual PoseCNN predictions on val RGB images,
followed by causal RGB-D LIP tracking. It does not use GT poses, GT masks,
FoundationPose, hand supervision, redetection, or GT resets during inference.
Object identity, CAD, and camera calibration are given. This is a known-object
tracking protocol, not class-agnostic object discovery.

## Initializer source and selection

The author-lab [PoseCNN repository](https://github.com/IRVLUTD/posecnn-pytorch)
links the public [checkpoint archive](https://utdallas.box.com/s/g0qnbs615kcizcqvys6pn96dr251m0lk).
`tools/fetch_posecnn_s0.py` downloads only its s0 epoch16 ZIP member using
validated HTTP ranges and verifies the ZIP CRC. The full 4 GB archive was not
downloaded or independently hash-verified. Extracted weight SHA-256:

`f2cc07f4e61b3077b76175459b5bace6405180e7efb70add115857d595d0a6a6`.

Upstream prediction code is pinned to
`f7d28f2abd38fcfc297d8d421bb6b69248626eb5`. A private build adapts the original
Hough voting and ROI pooling kernels to the installed PyTorch; a private Python
copy converts one unused target-helper scalar with `.item()` for NumPy 2.
The checkpoint loads strictly. Original preprocessing, CAD extents, quaternion
conversion, and class NMS are retained. PoseCNN's unused auxiliary target
arguments receive zero tensors; overlap computation aborts if reached.
Build, compatibility, and download receipts describe the exact changes.

For each stream, scan RGB frames in causal order from frame 0. Use the first
frame with any legal pose for the known object and choose the highest-confidence
candidate in that frame. There is no GT filtering and no additional confidence
threshold. Stop scanning when an initializer is found. The frozen manifest is
shared by every LIP candidate.

The actual full run found **319 initializers**, including **14 delayed** streams;
one stream had no legal initializer. There were 897 RGB attempts. Thus **578
frame positions have no pose** and stay in the primary denominator as failures.
No later pose is borrowed for an earlier frame. The initializer manifest SHA is
`ea33a4668526d2305f5a23d4e789d579094f0f2639ac81386c93766329e574bf`.

## Separate inference and scoring

`tools/infer_lip_val_non_gt.py` encodes the initial RGB-D observation and retains
the external pose, then updates its own history at every following frame. Invalid
proposals hold the preceding pose and advance the clock. Prediction JSONL hashes
are sealed before `tools/score_val_non_gt.py` can read GT poses for scoring.

The access guard rejects raw annotations, cached GT poses, other-split pixels,
and FP resources. It combines Python audit hooks with explicit checks before
native image reads; it is an auditable application guard, not an OS sandbox.

Primary scores are object-macro ADD@0.1d and ADD-S@0.05d over every frame,
including initial poses and missing-pose failures. Native-val visibility is
reused from a fixed full-val scoring artifact and never supplied to inference.
It is **not official BOP `visib_fract`**, and these threshold scores are **not
official test BOP AR**. Continuous errors are conditional on an emitted pose.
Bad-initial-pose groups are defined offline from the shared first initializer;
the first-eight population covers updates 1 through 8 after that initializer.

## Fixed candidate screening

The smooth rotation read uses `softsign(x) = x / (1 + |x|)` with derivative
`1 / (1 + |x|)^2`, preserving an exact zero start and allowing negative logits
to receive gradients. A negative coefficient extrapolates away from the initial
rotation. It is a signed residual coefficient, not a probability or a convex
mixture fraction. Center writing and the frozen core are preserved at equal
inputs/state. The architecture ID is `stream_rk_rotation_anchor_smooth`.

The control and smooth candidate share parent `19a63426…`, training draws,
startup-occlusion recipe, fresh optimizer, and fixed final **1,000 steps**. Only
that final step is eligible. The residual `89d5a66b…`, startup recipe
`603a59b9…`, and adaptive parent are also evaluated on the same initializers.
Historical checkpoints have different cumulative training budgets; only the
two new arms are a matched structural comparison.

`tools/compare_val_non_gt.py` uses 2,000 shared paired physical-sequence
bootstrap draws, seed 20260915. A new candidate must improve severe-occlusion
strict accuracy versus residual with a positive interval lower bound, retain
overall ADD and strict-ADD-S point scores, and not worsen first-eight bad-start
rotation/center point errors. Smooth additionally needs a positive severe
strict interval versus its matched control, an active read, and the same
overall/first-eight point guards versus that control. Otherwise residual is
retained. Point guards are not statistical noninferiority tests. The selector
and all inference/scoring dependencies were hash-bound before new full-val
scores were available. No official test launches automatically.

## Reproduction entrypoints and actual runtime

Use `--help` on `infer_posecnn_val.py`, `infer_lip_val_non_gt.py`,
`score_val_non_gt.py`, and `compare_val_non_gt.py` for standalone stages.
The orchestration command is:

```bash
/mnt/why/dexycb_lip/.venv-fp/bin/python tools/run_val_non_gt_screen.py \
  --spec runs/screen_spec.json
```

Run it from an isolated copy with a fresh `out` in the spec. Existing outputs
are never silently overwritten. The completed sealed runtime is
`/mnt/why/dexycb_lip/smooth_val_evaluation_20260915`; its spec SHA is
`710258c665b566d3cb7a748e12a4354a7c083949d05e0a4d679537225c568684`.
Training is in `/mnt/why/dexycb_lip/smooth_rotation_training_20260915`.
The selected checkpoint and complete selection receipt are produced only after
all required full-val runs finish. Runtime snapshots are not completion claims.

The completed run retained residual `89d5a66b…`. The smooth read received
gradients and stayed active, but did not improve on its matched control or the
residual baseline. Full-val strict ADD-S was 83.664% / 81.105% / 81.020% for
residual / control / smooth; bad-start first-eight center errors were 13.984 /
19.864 / 19.878 mm. See the [full report and paired intervals](../runs/smooth_completed_20260915/visual_review/report.md)
and [verification receipt](../runs/smooth_completed_20260915/completion/verification.json).
