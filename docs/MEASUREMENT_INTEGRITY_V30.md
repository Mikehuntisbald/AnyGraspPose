# V30–V31: completion weight and measured-surface integrity

Completed diagnostics, not new trained model versions. None of these interventions
has been promoted. Main restoration/pose accuracy still fails the original LIP
acceptance target. All use V21 step1000 and V29's fixed1000-step selector, the
same physical-sequence-disjoint controlled holdout, no optimizer updates.

## Oracle decomposition, explicitly unavailable at inference

V29 decomposition rejects the dense selector's RGB-invisible pixels using GT,
then replaces only the selected true measurements' canonical XYZ with
GT-inverse-transformed actual depth. JEPA predictions, appearance and readout
weights otherwise remain unchanged. This isolates errors rather than furnishing
a deployable branch.

V30 additionally defines a depth-consistency diagnostic: rendered GT depth is
positive and differs from measured depth by at most `max(10mm, 0.05*diameter)`.
This threshold is a declared diagnostic, not a proven sensor-error model. Disagreement
can include sensor noise, boundaries, alignment/annotation error, or CAD mismatch;
it is not by itself proof that the sensor point is wrong.

Among RGB-visible sensor points accepted by the dense selector,33.35% disagree
by this criterion in the non-symmetric ±10° cases, and41.68% in the artificially
occluded subset. Thus a visibility precision of78% does not establish78% correct
CAD-to-sensor correspondences. Future sensor-trust supervision must distinguish
RGB visibility, valid depth and geometric consistency.

|Controlled intervention|Non-symmetric rotation|Added-occlusion rotation|Zero drift|
|---|---:|---:|---:|
|Original|9.025°|9.341°|2.089°|
|Learned dense ownership|9.126°|9.797°|2.717°|
|GT removes RGB-invisible measurements|8.842°|9.204°|2.073°|
|Above + correct measured canonical XYZ|8.203°|8.604°|0.788°|
|GT also filters depth disagreements + correct XYZ|8.599°|8.978°|1.170°|
|Above, with completion weight set to zero|7.472°|7.365°|0.000°|

The last four rows use unavailable GT in the readout AFTER the complete student
forward. They are not validation gains, and changing measured support also
changes what the head can infer. Correcting measured points alone does not
make the original head reliable; the remaining completion and readout also
contribute. Removing completion helps this ideal-measurement diagnostic but
does not imply removing completion would help the actual model.

The initial generated oracle worker receipts inherited
`teacher_geometry_input=false` from the non-oracle script. This metadata field
was incorrect. Original receipts/results/source archives are preserved;
`provenance_correction.json` explicitly records oracle GT readout use. Committed
tools correct that declaration for future runs. The student forward never
receives GT features or geometry; no teacher encoder forward occurs.

## V31 deployable-input aggregate mass control

The existing0.5 per-pixel completion confidence does not cap TOTAL completion
influence. A large predicted area can overwhelm a few reliable measured points.
An optional `pack_completion(completion_mass_ratio=1.)` caps aggregate completed
weight at measured weight when any measurement exists, preserving the no-depth
RGB/CAD path otherwise. It has no GT input or new learned parameter and is
disabled by default.

|Input ownership|Non-symmetric rotation|Added-occlusion rotation|Zero drift|
|---|---:|---:|---:|
|Original|9.025°|9.341°|2.089°|
|Original + mass cap|9.006°|9.301°|2.166°|
|Dense|9.126°|9.797°|2.717°|
|Dense + mass cap|9.074°|9.702°|2.951°|

There is no convincing joint rotation/translation/zero-drift gain. Reweighting
also amplifies accepted false correspondences; this is not a validated fix.
Do not start a long training run merely because the confidence wiring is sound.
The next implementation must train/calibrate **correspondence reliability** and
the pose response on actual restored points, retain explicit measured versus
completed ownership, and verify matched pose/restoration gains before promotion.
Simply retraining a feature-only head, swapping a visibility classifier, or
imposing a global confidence cap is insufficient on these fixed probes.

Eleven targeted tests pass. Source archives, records, receipts and comparisons
are synced locally and remotely under report/artifact suffixes
`dense_measurement_decomposition_v29`, `measurement_integrity_v30`, and
`completion_mass_v31`. No package changes, history training, additional seed,
official test, or default-model replacement occurred.
