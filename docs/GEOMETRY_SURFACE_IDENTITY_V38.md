# V38: surface identity, real depth and mandatory CAD correspondence

V37 establishes small-sample learnability but poor generalization. V34 also
measured9.58mm mean real-target/GT-CAD depth disagreement. The previous canonical
head was asked to reproduce sensor-derived off-surface coordinates in artificial
occlusion and CAD surface coordinates in natural occlusion. This is a plausible
learning obstacle, not yet a proven sole cause.

This paired200-update test changes the interpretation of canonical XYZ, while
PRESERVING original real-depth supervision. It does not overwrite original
teacher targets, remove difficult evaluation pixels, or count a changed target
definition as an improvement against historical raw-XYZ metrics.

Both arms use the same V36 INITIAL tensors,seed42,8H20,batch32, full-window
single-frame examples and training seeds38000000. Source SHA256
`6bf4c1b03e68c8814373b2ce6834574cd3476e7053cf552df7759d4923edd6cc`.
The encoder/JEPA/DPT and transport head jointly train; pose, feature decoders and
history remain frozen. No DINO or pose loss. Learning rates: encoder1e-6,
representation3e-5,DPT1e-4,transport1e-3;20-step warmup,cosine floor0.5.

Shared changes in BOTH arms:

- The decoded XYZ equals the reference CAD XYZ at the predicted correspondence
  whenever that reference lookup is available. The free XYZ offset is disabled;
  low predicted confidence cannot choose the old DPT instead. Unavailable CAD
  retains the DPT fallback. Predicted support remains separately reported and
  is not a guarantee of correct correspondence.
- The existing real/proxy depth correction and depth targets remain intact.
- Direct flow loss weight12.5 replaces0.25, motivated by the measured50.9x
  other/direct gradient ratio. This is identical in both arms; no weight search.

The raw control retains sensor-derived canonical XYZ and full XYZ/depth
consistency on real regions. The canonical candidate supervises CAD surface
XYZ on those SAME regions and uses ray reprojection consistency there, allowing
real sensor depth to differ from the CAD surface. Natural-hidden proxy regions
retain full XYZ/depth consistency. Visible canonical correspondence is also CAD
surface identity where the GT CAD exists; visible RGB/depth reconstruction is
still absent. CAD normals and flow labels follow the corresponding XYZ contract.

The teacher exposes separate CAD XYZ/depth/validity metadata; existing mixed
real/proxy XYZ and real depth are unchanged. These fields never enter student
inputs. Unit tests verify preservation of real-depth supervision, sensitivity
to that depth target, independence from the legacy raw XYZ for the new canonical
loss, mandatory lookup, CAD dropout, and old teacher/decoder behavior.

Evaluate identical64 training-partition physical-holdout records at0/200. Report
historical raw-sensor XYZ AND CAD-canonical XYZ, raw-real/proxy depth, old flow
AND CAD-identity flow, with each eligibility count and identity baseline. The
native/fixed40 protocol is separate; no official-test use or model promotion.
Checkpoint every50 and exact2→200 resume. No automatic continuation after200.

Runtime`/tmp/dexycb_geometry_surface_identity_v38`; artifacts
`/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_surface_identity_v38`.

## Completed200-update comparison: no stable gain

Both arms completed200 updates and exact2-step resumes. CPU comparison verifies
identical starting model/optimizer and unchanged pose/feature heads. Eleven
targeted target-contract/decoder/teacher tests pass.

Equal-physical-sequence heavy metrics:

|Arm|Real raw XYZ mm|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|---:|
|Shared initialization|14.961|9.792|11.522|15.321|13.081|
|Raw-label200|14.762|9.482|9.871|14.996|13.473|
|CAD-label200|15.558|10.322|10.699|15.165|13.075|

Separating surface identity from real depth is NOT established as a sufficient
fix: the candidate loses to the raw control on real canonical XYZ and does not
improve proxy geometry meaningfully. No model promotion or longer continuation.
The metadata separation is explicit and tested, but the accuracy objective
remains unmet. V39 next tests whether current visible information is being used,
with frozen LIP and clean-input controls, before further decoder changes.
