# Observation reliability × keyframe memory

## Completed result (2026-09-14)

All four arms finished 1,000 steps and the same 320-stream / 23,200-frame
s0 validation. Tracking metrics exclude the 320 initialization frames.
The full paired report and predictions are in
`runs/rk_2x2_20260914_evaluation/` (231 files SHA-256 verified at collection).
The [factorial figure](../runs/rk_2x2_20260914_figures/rk_factorial.pdf)
shows the arm estimates, average main effects, and interaction intervals.
PNG/SVG exports, exact plotted values, source script and hash receipt are
in `runs/rk_2x2_20260914_figures/`; these render the completed report without
recomputing or selecting new scores.

| Object-macro success (%) | R0K0 | R1K0 | R0K1 | R1K1 |
|---|---:|---:|---:|---:|
| Overall ADD@0.1d | 79.855 | 79.847 | 79.889 | 79.948 |
| Visibility <0.5 ADD@0.1d | 58.192 | 58.257 | 58.810 | 59.904 |
| Visibility <0.3 ADD@0.1d | 41.826 | 41.925 | 42.457 | 42.742 |
| Visibility <0.3 ADD-S@0.05d | 47.782 | 47.978 | 47.957 | 47.940 |

Joint R1K1 improves visibility <0.5 ADD by 1.712 percentage points over the
matched R0K0 continuation, but only 1.178 points over the untouched parent
(95% paired interval [0.337, 1.504]). The R×K interaction is +1.029 points,
[-0.193, 1.550], so synergy is not established. The strictest occlusion metric
shows little improvement. This is one training seed and controlled noisy-GT
first-pose initialization, followed by independent causal prediction with
zero FP calls. These numbers are neither BOP AR nor a non-GT initialization
result. The original official test model remains preserved.

## Fixed experiment contract

This experiment tests independent streaming LIP on official s0 train/val. There
are zero FoundationPose calls, no MANO or hand-label supervision, and no new
test-set selection. The selected residual step-1000 parent is fixed by SHA-256
`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`.

| Arm | Observation weighting R | Additional keyframe memory K |
|---|---|---|
| R0K0 | Disabled | Disabled |
| R1K0 | Enabled | Disabled |
| R0K1 | Disabled | Enabled |
| R1K1 | Enabled | Enabled |

All arms retain the original eight-frame source/context cache and adapt the
existing cross-attention branch and its gate. The established RGB, geometry,
spatial/temporal encoders, original fusion, and pose head remain fixed. Thus
R0K0 is a matched training baseline, not simply an unevaluated parent number.
New residual paths start at zero. Parent output equivalence is checked with
nonzero trained weights, real training RGB-D fragments, and FP32/BF16 closed
loops before training. Weight identity alone is insufficient.

R uses a fixed geometry-support proxy from observed/rendered depth agreement
inside the rendered silhouette. For valid paired depths,
`q = 0.25 + 0.75 exp(-abs(depth_residual / diameter) / 0.05)`; unknown support
is 0.5. Sixteen pooled token scores bias attention and the pooled frame score
modulates the pose correction. Two learned scalar strengths start at zero.
The correction multiplier stays between 0.5 and 1.5, allowing damping or
amplification. This is a geometric proxy, not a calibrated uncertainty model.
It cannot disambiguate depth mismatch caused by occlusion from a wrong prior.

K adds four quality-ranked context anchors with minimum spacing four frames
and maximum age 64 frames. Only anchors older than the recent eight-frame
window can be read. Selection is causal, per stream, and uses the **same fixed
support proxy in both K-enabled arms**. R controls how that proxy weights
attention/corrections; it does not change K's admission policy. Nearby anchors
are replaced only for a support improvement above 0.05, preventing a sequence
of equally good frames from continually erasing the only older anchor.
Stored source context, timestamp, pose, crop intrinsics, and source coordinates
remain immutable. The new gated attention output and time residual start at
zero, preserving the parent's function.

This first factorial test treats R as a combined attention/update weighting
factor, and K as a combined quality selection/extra memory factor. It does not
separate the two components within either factor. All three learnable arms
add different compute; equal training samples/steps are not equal FLOPs.

The fixed budget is seed 42, 1,000 optimizer steps, 64 fragments per step,
eight burn-in plus 32 supervised frames, and two H20 GPUs per arm. Four arms
run concurrently on eight GPUs, with 32 fragments per GPU and no gradient
accumulation. The 64,000-entry sampling manifest is shared,
including initial-pose noise seeds. All arms use the same existing-branch LR
`1e-5`, new-module LR `5e-5`, quality-scalar LR `1e-3`, 100-step warmup and cosine
decay. Only enabled parameters receive updates. Final step 1,000 is declared
in advance; no best-on-test checkpoint is selected.

Preflight includes targeted unit/regression tests, real-fragment parent
equivalence, two full-unroll single-GPU steps, three two-GPU DDP steps and one
strict resume step for each arm. These are incremental architecture checks;
they are not represented as a new 300-step overfit result. Preflight artifacts
and formal training outputs are separate. The launcher fails closed if a
child fails or source/data/config hashes differ.

The validation target is the same 320 streams / 23,200 frames for every arm,
excluding initialization from tracking metrics. A fixed initial-pose manifest
with explicit provenance and split/mesh hashes is mandatory for this
comparison. Available PoseCNN predictions cover test only. For the subsequent
continuous train/val optimization request, controlled noisy-GT val initialization
was explicitly adopted as the agent's mechanism-validation default, with no
claim that a reply to the earlier optional question had arrived. All four arms
and the untouched parent have now completed this same-initializer evaluation.
No GT-initialized result may be described as a non-GT benchmark. See
`docs/ONGOING_OCCLUSION_OPTIMIZATION.md` for completed results and limitations.

`tools/evaluate_rk_ablation.py` runs the four final weights on matched val
streams with an explicitly supplied initializer file. The report includes
object-macro ADD@0.1d, ADD-S@0.1d/0.05d, center/rotation errors, lost rate, and
visibility <0.5/<0.3 plus long (>8-frame) occlusion episodes. Visibility uses
the existing validation segmentation/silhouette proxy, not official BOP
`visib_fract`. Recovery counts expose right censoring and already-successful
episodes; different failure denominators must not be mistaken for paired
causal recovery effects. This report is not BOP AR.

`tools/compare_rk_ablation.py` computes conditional R/K effects, average main
effects and interaction `R1K1 - R1K0 - R0K1 + R0K0`. It uses 2,000 paired
bootstrap resamples of physical sequences, carrying cameras together and
sharing draws across arms and metrics. It writes `comparison.json`,
`paired_physical_sequences.csv`, and `report.md`. These intervals measure
validation-population variation for one training seed, not seed robustness.

The initial preflight hit `LSE is not correctly aligned (strideH)` in attention
backward. A minimal GPU reproducer identified the efficient attention path
when Q/K/V are frozen and only the additive bias requires gradients. Disabling
cuDNN alone did not fix this. `StreamLayer.residual` now uses math attention
for exactly the bias-only gradient case. Other paths retain normal backend
selection. A dedicated CUDA test checks the output and bias gradient against
the math reference. There are no driver/global CUDA changes. The two failed
preflights and their original sources remain archived separately.

Remote runtime: `/mnt/why/dexycb_lip/rk_factorial_20260913_v3`.
Experiment: `runs/rk_2x2`. Raw data and prior run folders remain read-only.

```bash
cd /mnt/why/dexycb_lip/rk_factorial_20260913_v3
PY=/mnt/why/dexycb_lip/.venv-fp/bin/python
# The venv contains required torch/rendering packages; the model never imports FP.
$PY tools/run_rk_ablation.py --out runs/rk_2x2
# After training and after choosing/preparing the explicit initializer:
$PY tools/evaluate_rk_ablation.py --out runs/rk_2x2 --initial-poses INITIALIZERS.json
```

The launcher has an exclusive lock and refuses duplicate launches. For an
interrupted formal arm, keep its config and world size and use strict resume:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  $PY -m torch.distributed.run --standalone --nproc_per_node=2 -m lip.train_stream \
  --config runs/rk_2x2/R0K0/config.yaml --resume runs/rk_2x2/R0K0/train/last.pt \
  --output runs/rk_2x2/R0K0/train --max-steps 1000 \
  --fixed-manifest runs/rk_2x2/training_samples.json \
  --data-root /mnt/why/dexycb_lip/cache/raw_full_20260910 \
  --index-root /mnt/why/dexycb_lip/cache/dexycb_s0
```

Use the corresponding arm paths and GPU pair for other arms. Do not resume
into a different arm, modify the source under an active run, or reuse
preflight checkpoints as formal training starts.

Verified launch snapshot, 2026-09-13 08:53 UTC: all four formal runs were
active at steps 99 / 90 / 91 / 84 respectively. Recent step times were
2.26 / 2.41 / 2.46 / 2.63 seconds. These are progress measurements, not
validation outcomes. Forty-five distinct tests passed (two other tests were
skipped): 43 in the main CPU suite, one CUDA regression, and one additional
online-cache correction/reset test. The two analysis tests were rerun after
report changes. All four real-fragment equivalence checks were bitwise equal
in FP32 and BF16; single-GPU, two-GPU DDP and strict resume checks passed.

Gradient-enabled training invokes the bias-only math fallback for R1, so
BF16 training forwards have small numerical differences at the initial step
(rank-0 mean center error differs by approximately 0.017 mm). This is recorded
in `initial_training_numerics.json`; it is distinct from the bitwise-equal
evaluation-mode parent checks. No claim of training-mode bitwise identity is
made. The experiment retains this numerical limitation.

The local receipt snapshot is `runs/rk_2x2_20260913/`; all 99 downloaded files
were SHA-256 verified. `progress_snapshot.json` and `local_hash_verification.json`
identify that launch snapshot. It predates completed training and validation;
the final evaluation snapshot is `runs/rk_2x2_20260914_evaluation/`.
