# V41: audited-supervision matched trial

The geometry-only stage completed100 optimizer updates on8 H20s, seed42, from the same V38 raw200 checkpoint as V40 corrupted. Initial model, AdamW, schedule, eight rank RNG states and sampler state are exact matches. The stage performed a complete2→100 resume. All original64-record probe masks remain unchanged, including pixels filtered out during training. Pose/feature heads remain frozen; JEPA, encoder, DPT and transport train; history stays disabled.

Training changes: explicit FP32 geometry projection/objective and teacher-only depth/CAD disagreement quarantine at30mm with a2px eligibility margin. The launcher now runs the supervision-quality regressions before updates. Runtime is immutable `/tmp/dexycb_geometry_supervision_v41`; artifact root `/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_supervision_v41`.

Heavy corrupted-input probe, equal physical-sequence mass:

|Model|Real CAD XYZ mm|Real sensor XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|---:|
|Source|9.482|14.762|9.871|14.996|13.473|
|V40 corrupted100|9.715|14.075|9.891|11.448|10.192|
|V41 corrected100|9.477|14.306|11.561|12.783|13.324|

V41 does not meet accurate-recovery requirements. It does not improve jointly over the matched V40 arm, and depth regresses versus source. No extension, full40 promotion evaluation or default-model change is justified by this result. Correctness fixes stay; success is not inferred from filtered training loss.

The completed V40 full40 evaluation also failed to generalize: proxy XYZ36.735→37.709mm and depth19.883→20.838mm. These numbers use native baseline-conditioned crops and cannot be compared directly with the controlled10deg short probe. `tools/report_geometry_validation_v40.py` verifies common target counts and aggregates exact paired cases; history is disabled.

A separate model-free CAD reference coverage audit used16 training frames/10 objects, controlled rotations of the estimated reference at a fixed observed crop, and at most512 target pixels per real/proxy region. Within3% of object diameter, current-view sampled CAD coverage of proxy targets is100% at10deg,87.0% at45deg,61.6% at90deg and47.6% at180deg. An8192-point full-surface sample covers100% in these cases, with mean nearest-surface sample spacing about1.13mm. These are teacher-only geometric availability checks, not learned retrieval results. Raster nearest-point distance is NOT a mathematical lower bound for bilinear interpolation.

Two failure modes are therefore distinguished: small-error cases have the required CAD surface yet the learned correspondence remains inaccurate; large-error cases additionally lose reference-surface coverage. The next architectural experiment should make complete CAD local correspondence directly readable and supervised from the JEPA decoder, avoiding a sole reliance on current-view2D flow. A new head still has to demonstrate accurate correspondence and held-out geometry; the oracle coverage does not prove it will learn.

A secondary paired audit of8 saved heavy examples also separates sensor/CAD disagreement strata without changing the primary metrics. In the <=10mm target-gap stratum, depth error is6.20mm at source,5.96mm for V40, and8.93mm for V41. Thus the depth regression cannot be attributed solely to retained outlier evaluation pixels in these examples. The10–30mm stratum improves, but it cannot substitute for the complete result.
