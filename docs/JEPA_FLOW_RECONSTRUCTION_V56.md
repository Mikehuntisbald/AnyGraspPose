# V56: flow aligns CAD evidence, recovered geometry refines flow

This is a bounded 200-update prototype initialized from V53 step5200
(`717b503de54724e7dad79fe667b9fd49fe392b898af2522267add59ee65bdb3d`).
The goal remains accurate local reconstruction. Pose and history stay disabled;
no PnP, pose shortcut, FP branch, new encoder, or default-model promotion.

## Source and extension

[GoTrack section3.1](https://arxiv.org/html/2506.07155v1) predicts template-to-input
flow and visibility, using frozen DINOv2, a two-branch transformer and DPT. Its
flow L1 loss applies to visible template correspondences. This experiment borrows
the explicit endpoint/visibility formulation. Mutual JEPA reconstruction feedback
and supervised amodal endpoints are our extension, not claims about GoTrack.

## Forward path

1. Existing observed/rendered RGB-D fusion and four JEPA blocks produce patches.
2. Select one actual rendered surface pixel nearest each of256 patch centers.
   Canonical XYZ is sampled from that pixel, not averaged across surfaces.
3. Shared descriptors compare rendered appearance with both current JEPA patches
   and immutable observed appearance. Global matching plus local soft argmax and
   a bounded14px residual predicts each template point's input-crop endpoint.
4. Forward soft splatting writes CAD appearance, canonical XYZ, estimated-reference
   depth and observation residuals onto the input patch grid. A shared JEPA block
   processes this evidence before the shared DPT reconstructs geometry.
5. Recovered canonical XYZ supplies a bounded geometric matching bias in round2.
   Recovered validity supplies detached trust capped at0.5. Original observation
   descriptors/RGB-D remain available; estimated template depth is not treated as
   target camera depth. The second flow updates JEPA again, then DPT/full8192 CAD
   retrieval produce the final reconstruction.

Both rounds share parameters. The transformer refinement subtracts its no-write
pass so disabling transport does not silently add transformer depth. There is no
teacher input and no feedback from frozen pose outputs. This first prototype has
256 template endpoints, not GoTrack's full-resolution dense flow. CAD surfaces
absent from the current rendering still have the existing complete CAD atlas path.

## Independent supervision and checks

- Endpoint labels: GT projection of each sampled CAD point through the exact crop
  camera. Reject self-hidden/ambiguous boundary points using rendered GT depth and
  canonical identity. Supervise visible, artificially-hidden-real and natural-hidden
  CAD-proxy regions separately; the last region has half weight.
- Source ownership stays unchanged: artificially hidden real sensor depth remains
  real supervision; proxy depth is GT CAD rendering. No inferred confidence mask
  selects geometry or endpoint losses.
- Keep V53 geometry, coarse geometry, validity and atlas objectives. Add first-round
  reconstruction supervision and0.05 times the two-stage endpoint/matching loss.
  CE excludes both estimated-coordinate prior and inferred-geometry bias.
- Test both forward dependencies and gradient directions, empty CAD behavior,
  source masks, pixel conventions, and teacher isolation. First training update
  also checks independent float64 native-to-crop target projection and records
  shared-projection geometry/flow gradient norms and cosine.

All V53 tensors are checked for exact transfer. Only the flow interaction is
randomly initialized; optimizer resets for the new architecture. Eight H20s,
seed42,32 observations/update with paired pose hypotheses; strict full-state resume
after2 updates. New-module LR1e-4; existing rates follow V53's declared category
peaks with a20-step warmup. Save complete state every50 steps.

## Evaluation boundary

64 fixed training-partition physical-holdout observations: V53, V56 initialization,
V56 final, same-final-weight no-recovery-feedback, same-final-weight no-flow-write.
Report canonical XYZ/depth and per-source flow endpoint error. Also compare32
fixed observations at60-degree reference error, because full native validation
previously exposed a large-error weakness. These are controlled GT-perturbation
diagnostics, not native pose or rollout accuracy. No budget extension or accuracy
claim follows automatically from tests or training loss.
