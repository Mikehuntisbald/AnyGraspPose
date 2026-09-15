# Independent LIP: comparison scope and improvement priorities

2026-09-15 update: the matched rotation-anchor retraining, full controlled val,
and frozen standalone official tests have completed. The added anchor read is
zero on all 22,880 tracked val frames; a read-only 112-observation train probe
confirmed negative logits and zero gradients through the hard clamp. This
candidate was rejected by the predeclared rule. The retained S1O1 seed-42
checkpoint (`603a59b9...`), not the rejected rotation-anchor checkpoint, then
scored all/grasped AR 71.4294/68.5395 on the complete non-GT, zero-FP s0 test.
The same-entry residual-1000 rerun scored 72.2770/70.0599. Grasped visibility
<0.3 AR fell from 41.7529 to 38.8959 (difference -2.8570 pp, paired 95% interval
[-5.0106, -0.9575]). Thus the replicated controlled-val startup-occlusion gain
has not transferred to this official protocol; it is not a SOTA result.

See the 2026-09-15 section of `ONGOING_OCCLUSION_OPTIMIZATION.md`. Priorities are
to correct the gate's negative-logit gradient dead zone and validate non-GT
initialization on train/val before another frozen test. Split, initialization
and metric all differ between controlled val and official test; the current
comparison does not isolate which caused the regression. The observations and
literature comparisons below describe the original residual-1000 model at the
dated audit, not a fresh external-method ranking.

Audit date: 2026-09-13. No model changes or training were performed for this review.
The primary experiment is the completed residual-1000 `runs/lip_only_s0_test`,
with direct PoseCNN initialization, zero FP calls, causal RGB-D tracking, and
official s0 test BOP target scoring. All-object AR is 72.2770%; grasped AR is
70.0599%. The defensible position is a competitive independent temporal refiner,
with a positive measured history effect, rather than a demonstrated current SOTA.

## Comparisons

The official 2021 DexYCB supplementary Table 3 lists S0 grasped-object AR references:
DeepIM RGB-D 62.03, CosyPose RGB 59.99, and PoseRBPF RGB-D 49.09. These are useful
historical references, not a matched current-method ranking: modalities, temporal
inputs, initialization, recovery policies and training budgets must also align.
Our reproduction of the released DeepIM predictions gives grasped AR 62.0275.

Source: https://yuxng.github.io/Papers/2021/chao_cvpr21_supp.pdf

Visibility Aware In-Hand Object Pose Tracking in Videos With Transformers (2025)
reports DexYCB ADD(-S) AUC 80.1 overall and 78.3 for visibility <0.5. Its use of
past and future frames, hand-projection visibility supervision, and AUC differs
from our causal, hand-free, BOP-AR experiment. Those numbers do not establish a
ranking against LIP's 70.06 grasped AR or 52.41 heavy-occlusion AR.

Source: https://www.researchgate.net/publication/389299729_Visibility_Aware_In-Hand_Object_Pose_Tracking_in_Videos_with_Transformers

GenHOI (2026, current v4) uses hand priors/MANO and reports its DexYCB unseen-object
comparison on S3, including ADD-0.5D/AUC and ADD-S in mm. It is not our S0 RGB-D
causal object-only tracking protocol. RRTrack (July 2026 preprint) combines a
frozen FP refiner, memory-based segmentation and recovery templates; its main
quantitative tracking benchmark is synthetic. It motivates studying memory
validation and recovery but is not a same-protocol DexYCB SOTA result.

Sources: https://arxiv.org/html/2603.19013v4 ; https://arxiv.org/html/2607.23669v1

An existing separate matched-initialization control is also informative: after a
common PoseCNN→FP initializer, LIP-only tracking scores all/grasped AR
74.0310/71.6868 versus FP tracking 73.1553/61.1433. That control supports LIP's
tracking competitiveness but is not the primary zero-FP experiment and does not
match training data budgets. Do not mix it into the zero-FP table.

## Confirmed code/data observations

1. `stream_features.py:33–35` derives token role bias and its `reliable` flag from
   the rendered CAD silhouette. It does not explicitly determine whether the
   observed surface is occluded. The network DOES receive observed depth and
   depth residuals, so this does not prove it is blind to occlusion; the missing
   component is explicit, calibrated observation reliability for updates/memory.
   `stream_batch_features.py` implements the same training-side semantics.
2. The explicit source/context window is eight recent frames. No quality-selected
   long-term anchor bank exists. This is not a claim that the total recurrent or
   multilayer receptive field is only eight frames.
3. A new read-only diagnostic over 87,445 actual initialized grasped-object frame
   positions finds 736 visibility-<0.5 episodes, 383 longer than eight frames;
   88.99% of those low-visibility frame positions lie in episodes longer than
   eight frames. For <0.3, the figures are 386 episodes, 171 longer than eight,
   and 85.25%. These use full-rate BOP visibility including intermediate frames;
   they are not a new accuracy population. Raw results are in
   `runs/lip_only_s0_test/occlusion_episode_diagnostic.json`.
4. The current V2 training configuration uses eight burn-in and 16 supervised
   unroll frames. StreamClips samples objects/physical sequences/cameras and
   starts, without a dedicated occlusion-duration sampler; augmentation is false.
   This does not say previous training contained no naturally occluded images.
5. Spatial fusion is pooled to 4×4 visual tokens. Grasped per-object AR is weak
   on foam brick (36.1100), scissors (41.0862), and large marker (58.8896), compared
   with cracker box (90.1735). Pooling, geometry mismatch, symmetry and texture are
   hypotheses to separate, not established causes from these aggregate values.
   Current pose_loss penalizes a single GT rotation and point correspondence,
   without an explicit symmetry-equivalent target set.

## Prioritized experiments, all without FP or hand supervision

P1: Add a calibrated observation-quality signal from RGB-D evidence and CAD
agreement. Use it for soft weighting of pose correction and memory writes/reads.
Treat invalid depth as unknown, and distinguish occlusion from a wrong prior;
hard rejection of all depth disagreement can prevent recovery from a bad pose.
Use object-only training labels or pose-error supervision; no hand model is needed.

P2: Combine recent history with reliable pre-occlusion keyframes. Preserve source
crop/intrinsics/pose metadata and causal timestamps. Train with contiguous
occlusion episodes and longer 32/64-step closed-loop rollouts. Do not merely
increase memory_frames at evaluation: the checkpoint/cache contract and training
distribution must be updated consistently. Test recovered accuracy and recovery
latency after long occlusions, in addition to threshold-bucket AR.

P3: Investigate small/thin/ambiguous objects separately. A controlled 4×4 versus
8×8/local-detail token experiment and symmetry-aware loss are reasonable next
ablations after checking per-object depth/crop alignment. Existing geometry audits
already caution against treating every depth disagreement as a model error.

The first recommended study is a 2×2 ablation: observation-quality handling
off/on × reliable keyframe memory off/on, with matched initialization, compute
accounting and training schedules. Select designs on train/val; report frozen
test results and explicitly recognize that the current test has been inspected.
The proposed interventions have not yet demonstrated gains. The measured result
is only the existing full-history versus cleared-history effect (+1.82 and
+2.19 grasped AR points in the <0.5 and <0.3 subsets).
