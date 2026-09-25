# V24: geometrically grounded readout, then restoration through a fixed readout

Goal: CAD-assisted recovery of pose-useful local appearance and geometry, then
pose from actual restoration. Original native LIP acceptance remains unmet.
V23's cached CAD-coordinate decoder did not improve development geometry and is
not used here. The V21 restoration backbone and production default are preserved.

## Completed readout comparison

The V21 readout multiplies its learned delta by point-residual RMS/0.1d. A fixed
rotation angle produces different RMS for different object shapes/axes. Its
moment token also divides covariance only by total spread, leaving the network
to learn shape-dependent inversion. These are tested hypotheses, not an assertion
that every remaining pose failure comes from the head.

Three arms start with identical V21 step1000 shared readout parameters and train
600 steps on the same192 ideal-geometry packets, cached STUDENT appearance,
perturbations, sample order and AdamW3e-4.64 physical-sequence-disjoint training
records are the development set. No encoder/backbone is updated in this phase.

- Legacy: original token and RMS output multiplier.
- Unit gate: preserve the token; replace magnitude multiplication with a smooth
  evidence/zero-residual gate that rapidly saturates to1.
- Shape conditioned: unit gate plus an extra learned projection into the SAME
  object-query relation token. The six-dimensional input normalizes rotational
  torque with damped shape information`trace(S)I-S`, and centers translation.
  Its projected features do not directly update pose. No raw JEPA/FP/teacher
  shortcut exists in deployed forward. Small-angle normalization is bounded and
  damped on unobservable axes; the final delta still comes from the learned head.

|600-step readout|Development10-degree rotation remaining|Signed response gain|
|---|---:|---:|
|Legacy|2.757 deg|0.819|
|Unit gate|2.332 deg|0.941|
|Shape conditioned|0.616 deg|1.008|

Controlled validation rerenders actual frames and uses the same476 cases from40
sequences as V22. The following means use249 positive-axis10-degree cases from28
non-symmetric sequences. Ideal-geometry replacements are NOT deployed scores.

|Readout|Actual JEPA geometry|Ideal CAD geometry|Sensor-consistent ideal geometry|
|---|---:|---:|---:|
|Original V21 joint-trained head|9.434 deg|7.733 deg|7.857 deg|
|Legacy,600 oracle updates|26.649 deg|4.291 deg|7.080 deg|
|Unit gate,600 oracle updates|18.457 deg|2.673 deg|4.076 deg|
|Shape conditioned,600 oracle updates|21.163 deg|0.686 deg|1.093 deg|

On54 heavy cases from9 sequences, the shape-conditioned head gives25.191 degrees
with predicted geometry and0.716 degrees with ideal CAD geometry. Thus correct
geometry is readable, but more responsive readout amplifies current restoration
errors. The original head's attenuation can partly be an adaptation to noisy
inputs. None of these readout-only checkpoints is promoted. Ideal-cache retention
alone is explicitly insufficient evidence of native improvement.

## Bounded restoration training now running

Run500 additional restoration updates, source step1000 to1500, seed42,
8H20/effective batch32, history off. The shape-conditioned readout above is
FROZEN; gradients still pass through it to decoded DINO features, XYZ/depth and
shared JEPA patches. This tests whether a calibrated readout can teach geometrically
useful restoration without adapting itself to suppress incorrect geometry.

Restoration/EMA tensors and420 existing Adam states are restored exactly from
V21 step1000; readout tensors are explicitly replaced and excluded from Adam.
Unused source parameters without Adam state are listed in the adaptation receipt.
All eight rank RNG states are restored. The declared500-step LR horizon warms
from0.3 of peak over50 updates, then decays to0.5 of peak. Existing peak LRs and
scalar restoration/pose losses are unchanged. Oracle rehearsal is disabled
because the readout is frozen. Paired RGB-D remains identical; feedback motion
state uses the previous estimated pose and timestamp, as in the V22 common fix.

The eight-GPU preflight passed: pose gradients reach XYZ, depth, both recovered
feature layers and shared patch (norms0.0487,0.0404,0.000511,0.000486,0.00753).
Full checkpoint continuation1002→1003 is exercised before the main run. Full
native validation follows1250 and1500; fixed40 reconstruction and controlled pose
probe follow1500. The controller does not exceed1500 or switch the default model.
Geometry and native pose must both be checked; reducing training pose loss or
passing the ideal readout gate is not task completion.

Runtime `/tmp/dexycb_shape_conditioned_v24_train`; artifacts
`/mnt/why/dexycb_lip/unified_jepa_20260921/shape_conditioned_v24/grounded_recovery`.
Readout comparison runtime `/tmp/dexycb_shape_conditioned_v24` is separate.
Validation aggregation initially lacked a copied reporting script; completed
inference shards were verified and reused after the script was copied, not rerun
or overwritten. Training source snapshots remain pinned and separately archived.

Validation so far:15 tests passed (shape/sign/degeneracy, legacy identity, exact
named Adam migration, serial restoration gradients, geometry-priority helpers
and pose geometry), plus the actual eight-GPU preflight and strict resume.
