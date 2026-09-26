# V39: does removing artificial occlusion improve actual correspondence?

The same64 training-partition physical-holdout cases are evaluated twice with
frozen V38 canonical200 and the original pure LIP checkpoint
`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`.
Both models receive the same estimated pose,224px crop and corrupted RGB/geometry
tensors. LIP has empty history and performs ONE refinement. Its resulting CAD
render is a diagnostic reference; no LIP output enters JEPA or training.

The clean control removes artificial corruption from the current input only,
while preserving original target regions. Thus it restores available sensor
evidence without feeding a GT render or GT feature. It is a privileged
input-availability control, not deployable heavy-occlusion accuracy.

GT noisy poses are controlled starting hypotheses, not native initialization.
This probe cannot be compared as a native ranking with the original23,200-frame
LIP score. Failure of a frozen baseline also cannot establish an information-
theoretic impossibility; its empty-history distribution is a limitation.

Equal-physical-sequence heavy LIP rotation error after the same10-degree start
is10.556degrees with artificial occlusion,5.916degrees with it removed. The
original visual input therefore enables a substantially better response from
this frozen reference under the controlled protocol.

JEPA's ALL-target CAD canonical XYZ and CAD-flow EPE do not improve when the
artificial occlusion is removed (record-mean real XYZ10.803→11.103mm,
proxy14.916→15.229mm; real EPE5.793→5.880px,proxy4.839→4.935px).
This is stronger evidence for inadequate use of visible correspondence cues
than for a decoder numerical failure. It does not prove the entire latent
contains no useful information or identify a unique architectural cause.

The report also gives rendered/decoded coverage and the fraction of ALL target
pixels with valid output,canonical XYZ<10mm AND depth<5mm. Missing surfaces count
as failures in that fraction. Conditional covered-pixel mean errors have
DIFFERENT pixel populations and must not be used as a whole-region ranking.
JEPA depth in this diagnostic is its decoder output, BEFORE the production
measurement override; clean-input depth numbers do not measure that routed
production path. Canonical XYZ/flow conclusions do not depend on the override.

Both8-worker runs and paired base-render/target identity checks completed.
No training, package changes, test-split access or default-model change.
Runtime copies are `/tmp/dexycb_geometry_lip_reference_v39` and its`_clean` peer.
Artifacts are under`/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_lip_reference_v39`.

Next bounded work: establish visible-surface correspondence first, using paired
estimated poses of identical observations, before asking the same representation
to recover heavy occlusion. Restoring heavy geometry remains part of the goal;
a successful clean-only curriculum would be a prerequisite, not completion.
