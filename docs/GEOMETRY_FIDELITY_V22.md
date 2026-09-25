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
