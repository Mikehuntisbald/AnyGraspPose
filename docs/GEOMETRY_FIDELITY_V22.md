# V22: preserve physical restoration while learning pose

The active goal remains CAD-assisted recovery of pose-useful missing appearance
and geometry, then pose through that recovery. V21's52.47% native ADD-S@0.05d
does not meet the old-LIP acceptance target; its geometry recovery regressed.
This stage must improve the actual restoration/pose outcomes, not just pass a
readout unit test or make a gradient nonzero.

## Frozen evidence from V21 step1000

64 training-split examples, each with a10-degree initial rotation perturbation
and a real-textured heavy RGB-D occluder, were audited without production updates.
At the shared final patch, pose/direct-delta gradient norm is7.51 times geometry
gradient norm at the median. Median cosine is0.087;17.19% conflict. XYZ and depth
median norm ratios are2.11 and1.55; conflict fractions14.06% and25%. This is
mostly magnitude/near-orthogonal pressure, not universal sign conflict.

Teacher canonical XYZ, camera transform and depth agree to below0.001mm mean
in the audited groups. There is no evidence here of a unit/frame mismatch.
Originally visible sensor depth differs from same-view CAD rendering by10.90mm
on average;1.92% of these eligible pixels differ by over50mm. Those values remain
real measurement targets; this experiment does not silently replace them by CAD.
Among examples with eligible hidden geometry,12.70% of real cases and44.44% of
proxy cases have no eligible DINO reconstruction patch under the existing90%
patch-purity rule. This boundary/thin-object coverage remains an open issue.

A separate decoder-only capacity probe froze JEPA tokens and coarse routing.
On46 fitting records,500 DPT updates reduced real XYZ14.57→7.35mm and depth
10.08→4.98mm. On18 physical-sequence-separated development records, XYZ worsened
19.92→23.83mm. The DPT can fit the provided representation on a small set; this
does not establish generalizable restoration, and its weights are not deployed.

## Paired continuation experiment

Both arms start with exact V21 step1000 weights, EMA, Adam moments and eight-rank
RNG (SHA256`c353d5bda97ff33782c65b5adde8e492303ef6ea30e36c30f514e4e6b1d77867`).
Use seed42,8H20, effective batch32 and the same scalar losses. The same explicit
continuation schedule preserves the0.3 boundary LR factor, warms to peak over50
updates, and defines a1000-update continuation horizon. The initial paired pilot
ends at1200:200 additional updates per arm, not two random seeds.

- Control: sum the unchanged scalar geometry/appearance/pose objectives normally.
- Priority: estimate geometry and secondary (appearance+pose) gradient norms at
  the shared patch. During secondary backward, scale only restoration parameters
  by`min(1, 0.5 * geometry_norm / secondary_norm)`. Pose-readout and decoded-feature
  output heads keep their full gradients. Then add the full geometry gradient.
  The forward tensors and deployed inference graph are unchanged. This is not a
  mathematical guarantee about a finite Adam step or a claim of measured gain.

Both arms retain head-only oracle rehearsal, exact paired RGB-D and geometry
targets. They also share a state-distribution repair: the feedback frame now
receives the previous **estimated** pose and timestamp, matching native runtime.
The former trainer updated its crop from the student but left these motion-state
fields empty. Paired hypotheses retain the same motion state so it cannot reveal
the synthetic perturbation. Thus comparison against V21 includes this common
repair; comparison between the two arms isolates gradient-budget handling.

## Evaluation and boundaries

After each pilot: full native23,200-frame validation and fixed40 reconstruction
with the same pure-LIP crop reference, donor augmentations and fixed DINO teacher.
Evaluate real/proxy XYZ and depth separately, local feature/CAD correspondence,
and all/heavy/extreme native pose. Favorable training loss alone cannot select
the priority arm. No default-model replacement, official test, new environment,
FP bypass, or raw-patch pose bypass is introduced. The original user goal stays
active if these outcomes remain insufficient.

Artifacts: `/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_fidelity_v22`.
Execution: `/tmp/dexycb_geometry_fidelity_v22_train` (pinned source snapshot).
The controller serializes training and evaluation across all8 GPUs. A new
continuation is not automatically launched merely because the pilot finishes.

## Completed pilot: reject gradient-priority continuation

The controller completed both200-update arms and all native/recovery evaluations.
CPU tensor hashing verified equal starting model, Adam, scheduler and eight-rank
RNG; normalized configurations differ only in output path and enable flag.
Native all/heavy/extreme ADD-S@0.05d: control50.93/28.73/7.58%, priority
48.65/23.82/2.74%. Priority improves some geometry errors by3–4% relative to
control but worsens CAD-proxy depth by6.58%. This intervention is not adopted.
No long continuation or default replacement follows this failed pilot.

## Current-pose CAD geometry audit

Using the V21 step1000 model, fixed40 sequences and exactly the previous recovery
protocol, compare raw estimated-pose CAD geometry against decoded geometry.
GT is used only for evaluation targets. Scores below restrict both methods to
the SAME target pixels also covered by the estimated CAD render; uncovered
pixels remain a separate failure and are not silently excluded from full scores.

| Heavy target | CAD coverage | CAD depth mm | JEPA depth mm | CAD XYZ mm | JEPA XYZ mm |
|---|---:|---:|---:|---:|---:|
| Real sensor |93.36%|10.98|16.36|37.49|37.32|
| CAD proxy |83.13%|16.03|24.12|37.32|38.28|

Whole-target raw-CAD depth error, assigning its absent depth zero as stored, is
60.68/143.25mm. Thus the overlap advantage does NOT establish a usable complete
surface or better pose. Non-symmetric objects alone retain the depth gap:
8.12 vs13.89mm real,14.62 vs23.22mm proxy. Symmetry cannot explain it away.
The next diagnostic independently replaces canonical XYZ and missing depth,
keeping confidence/support and measured depth fixed, to identify which geometric
error limits the actual learned pose head. GT replacements are diagnostic only.

Receipts and numerical summaries:
`reports/jepa_20260921/unified_rgbd_v2/geometry_fidelity_v22/`.

## Geometry/readout isolation (completed, frozen V21)

The final component probe uses476 cases across40 validation sequences. Rotation
numbers below use249 positive-axis10-degree cases from28 non-symmetric sequences;
heavy is54 cases from9 sequences. These are controlled interventions, not native
accuracy or training outcomes. Every condition reuses the same frame and head.

| Intervention | Non-sym rotation deg | Heavy rotation deg |
|---|---:|---:|
| Current learned readout |9.434|10.429|
| Correct missing depth; fixed support/measurements |9.387|10.098|
| Correct canonical XYZ; fixed support |8.562|10.419|
| Correct both; fixed support/measurements |8.450|9.484|
| Correct sensor-consistent XYZ/depth, pruned support |7.857|7.913|
| GT-error-dependent confidence only |8.734|8.918|
| Robust rigid fit of actual predicted correspondences |17.753|31.068|
| Rigid fit of sensor-consistent ideal correspondences |0.00021|0.00031|

Sensor-consistent oracle means visible measured points are assigned their
canonical coordinates through the GT transform; missing points use GT CAD.
This avoids treating the10.9mm average sensor/render depth disagreement as a
coordinate-system defect. With a correct initial pose this packet gives the
learned head0.00034-degree output, while it still under-corrects10-degree errors.
The readout therefore has a nontrivial limitation even with correct geometry;
its training-cache oracle score must not be substituted for this validation result.

Current weights outside GT silhouette average12.81% (heavy25.91%). Weights
mistakenly owned by measured object pixels average13.89% (heavy34.17%), expressed
as fractions of ALL relation weight, not measured weight. The categories overlap.
Simply pruning GT background does not improve this fixed head, so contamination
alone is not established as the entire cause. GT-quality weighting retains33.01%
of total weight (heavy20.56%) and improves the learned pose but still falls short.
A learned confidence head is a candidate, not an already-verified fix.

The rigid fit is an independent solvability diagnostic, NOT a deployed SVD/PnP
replacement. A known-transform synthetic test checked its rotation/translation
conventions. Its predicted-input failure rejects the idea that replacing only
the learned head with a solver solves this experiment. Both correspondence
fidelity and the readout's response to geometric evidence require improvement.
Evidence: `geometry_quality.json` and remote `geometry_quality/rank*/frames.jsonl`.
