# V36: explicit reference geometry in the correspondence decoder

V35's frozen decoder did not learn useful flow: heavy real/proxy EPE5.897/5.009px
versus identity5.219/4.463px. Geometry also regressed against its own changed-gate
initialization (real XYZ13.708→13.796mm, proxy15.283→15.996mm). All699 existing
non-transport tensors were verified unchanged. The accuracy goal remains unmet.

The small V36 test adds nine explicit reference channels to the transport head:
estimated-CAD canonical XYZ/depth/validity and the signed difference between
the decoded XYZ/depth query and that reference. CAD dropout zeros every new
channel. The flow query still originates in the unified JEPA/DPT; no FP/RGB
encoder branch or alternative pose prediction is added. The fixed reference
already provided lookup values in V34; it now also conditions WHERE to read.

Parent is V35 INITIAL checkpoint (before its unsuccessful100 updates), SHA256
`3116e0bdc03313983ad6a5599d8e4c55e4b66cd9f86420f53137f8a80c125837`.
The first convolution expands16→25 channels, retaining every old coefficient
and initializing all nine new channel weights tozero. Every other tensor is
exact. This preserves the mathematical initial function; small floating-point
kernel differences must not be presented as bitwise forward identity.

Only the transport head trains for100 updates. Optimizer, schedule, seed42,
8H20,batch32, training seeds, targets, and fixed64 physical-holdout records match
V35. All pre-existing encoder/JEPA/DPT/EMA/pose tensors remain frozen. Verify
the new-channel gradient, exact frozen tensors and2→100 complete resume.
Evaluate changed-model step0 and step100, zero-flow and reference-overlap controls.
No automatic longer continuation, test-split access or default-model promotion.

Separately, an analytical projection-only audit of V34's untrained source maps
its decoded canonical XYZ through the estimated pose into the CAD raster. It
improves proxy depth but worsens real XYZ; predicted reprojection errors remain
worse than identity. This rejects treating surface snapping alone as accurate
correspondence recovery. Its metrics use all eligible target pixels, with
original predictions retained where the reference cannot be read.

Runtime`/tmp/dexycb_geometry_transport_conditioned_v36`; output root
`/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_transport_conditioned_v36`.
