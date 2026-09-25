# V23 bounded CAD-coordinate decoder pilot

Status: experimental decoder implemented, tests and500-step comparison completed;
development improvement gate failed. It is not part of the deployed V21 graph and
has no native pose result yet. The original restoration/pose goal remains unmet.

V22 isolated two shortcomings: actual recovered point correspondences fail a
robust rigid-fit diagnostic, and the learned head under-corrects even ideal
geometry. Changing gradient weighting failed native validation. This pilot first
addresses canonical correspondence fidelity, preserving the old experiments.

## Decoder

Four frozen JEPA block outputs enter the existing DPT pyramid. Its dense feature
map supplies56×56 learned queries.256 actual complete CAD surface points provide
keys from their frozen Utonia descriptors and dynamic camera geometry; canonical
CAD coordinates are the values. A soft projected-UV prior helps matching but
never enforces same-pixel correspondence. The retrieved coordinate field is
upsampled and receives a bounded0.1-diameter local offset predicted by DPT.
The prior DPT depth/validity outputs remain. Missing CAD reference falls back to
the old DPT output exactly; invalid reference values cannot leak NaNs.

This is a prototype readout inside geometry decoding, not an FP/pose shortcut.
No pose head has been connected to it yet. Soft retrieval can still average
multiple surfaces; nearest-anchor CE explicitly tests whether correspondence
learning prevents that failure. Symmetry representatives are NOT changed per
pixel: all labels remain in the original texture-aware canonical frame.

## Bounded comparison

- Source V21 step1000 SHA256c353d5bda97ff33782c65b5adde8e492303ef6ea30e36c30f514e4e6b1d77867.
-256 training-split frames, heavy real-textured RGB-D occlusion, ±10-degree bases.
- Fixed JEPA and coarse-routing cache; physical-sequence-disjoint development
  split within training data (SHA256 first8 modulo4 equals0).
- Same seed42, sample order,500 updates, batch8 and AdamW3e-4 for each arm.
- Control: existing DPT fine-tuned on geometry/visible-correspondence/consistency.
- CAD arm: same losses plus0.1 dense nearest-anchor CE. This tests the combined
  representation+supervision change; it does not isolate attention architecture.
- Full native validation, pose improvements, and history gains are not inferred
  from this cache experiment. A positive development result is only a gate for
  integration and end-to-end testing; a negative result is not extended blindly.

The cache uses eight GPUs. Two small independent decoder fits use one GPU each;
this is a diagnostic, not an eight-GPU formal continuation. No old run is deleted,
no default checkpoint changes, and no official test or multi-seed run is used.

Runtime `/tmp/dexycb_cad_coordinate_v23`; artifacts
`/mnt/why/dexycb_lip/unified_jepa_20260921/cad_coordinate_v23_pilot`.
The controller pins source hashes and saves source, cache and terminal receipts.
Validation:13 tests passed, including CAD-query/backbone gradient flow, empty-CAD
fallback, canonical anchor labels, DPT behavior and original serial readout.

## Completed result: do not adopt this prototype

Both500-step fits finished. There were190 fitting records from164 physical
sequences and66 development records from53 disjoint physical sequences.
Posthoc metrics below exclude empty target frames and average physical sequences;
the saved fit curves use the earlier batch masking convention and are retained
unmodified. Neither those curves nor this table is a native validation metric.

| Decoder | Real XYZ mm | Real depth mm | Proxy XYZ mm | Proxy depth mm | Visible XYZ mm |
|---|---:|---:|---:|---:|---:|
| Parent, no fitting |15.528|11.696|15.141|15.699|21.030|
| DPT,500 updates |16.954|11.695|17.685|13.102|21.995|
| CAD readout,500 updates |18.473|11.761|17.258|15.114|24.393|

Real metrics cover64 frames/52 sequences; proxy24 frames/19 sequences;
visible65 frames/52 sequences. The CAD prototype does not pass the development
improvement gate. It is preserved as experimental code and is not integrated
into the tracker or promoted to a default. The frozen-backbone, small-data result
does not prove that a jointly trained CAD-conditioned decoder cannot work.
It rejects treating this particular500-update prototype as an achieved repair.
All cache/fit processes terminated successfully; production weights unchanged.
