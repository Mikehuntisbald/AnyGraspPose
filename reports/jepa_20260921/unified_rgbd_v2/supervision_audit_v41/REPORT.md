# V41 supervision audit — completed checks and remaining limits

This audit ran without loading a learned model or optimizer. It checked 16 fixed training frames (seeds 40000000–40000015), 10 objects, with real textured synthetic occlusion. No held-out frames were used. No new training has been launched by this audit.

Two actionable problems were found:

1. The flow teacher inherited BF16 autocast from the geometry objective. For an exact same-view plane, a zero-flow target had a maximum 0.25 px error under AMP versus 0.00000763 px in FP32. `transport_targets` and the geometry objective now explicitly disable CUDA autocast. Before/after receipts bind the tested source hashes. All target outputs now match bitwise under outer AMP.
2. The one-diameter real-depth plausibility guard admitted RGB-visible pixels whose sensor depth disagreed with the GT CAD by up to 156.035 mm. Inspection of case03 places its largest errors near the object boundary. This establishes unreliable measurement/annotation agreement, not which sensor or label is at fault. New optional teacher-only filtering excludes >30 mm conflicts plus a two-pixel eligibility boundary. It marks conflicting validity labels unknown, never replaces real targets with CAD, and also filters the separately supervised visible correspondence path.

The new config `configs/jepa/geometry_supervision_v41.yaml` enables this policy. It preserves V40 corrupted's source checkpoint, sampler and 100-update budget for a matched follow-up. Historical configs, checkpoints and scoring masks are unchanged. The 30 mm threshold is a declared conservative policy, not an estimated sensor noise bound or proof all retained labels are correct.

Verified construction checks:

- Native JPG RGB, aligned depth PNG, segmentation NPZ and original pose labels equal the training input/cache values exactly for every sampled frame.
- Independent double-precision linear solves recover the same canonical XYZ from real depth, GT rotation/translation and crop K. Centered/original pose conversion is also checked analytically.
- A rotated analytic plane, translated camera, asymmetric intrinsics and nonidentity crop verify the renderer against ray–plane intersection, independently of the renderer implementation. Maximum interior depth error is 0.0001375 mm; maximum all-raster plane depth error is 0.03953 mm.
- Pixel-center and nearest-crop indexing are checked by independently computed integer source indices. There is no observed half-pixel, unit or rotation-transpose mismatch.
- Real reconstruction masks are a subset of original-visible AND synthetic-hidden; CAD proxy masks are naturally invisible silhouette regions. They do not overlap. Unchanged-visible pixels have no reconstruction targets; visible correspondence is a separate explicit objective.
- Real target depth is exactly original unaugmented sensor depth; proxy depth/XYZ come from GT CAD. Invalid sensor depth does not get replaced by CAD. Donor RGB-D is unchanged outside its mask.
- For legacy mixed real/proxy geometry targets, maximum supervised XYZ-to-depth inconsistency was 0.000112 mm. CAD-canonical XYZ and real sensor depth are intentionally separate definitions; the canonical objective uses ray consistency for real targets, full camera consistency only for proxy targets. An exact-target gradient regression test verifies no contradictory XYZ/depth pull at the optimum.
- Building an alternate GT teacher leaves the estimated-pose student render unchanged. This audit does not claim a new whole-network forward-isolation test; previous model isolation tests remain distinct evidence.
- All 2,958 proxy pixels were unchanged by filtering; none had valid original sensor depth more than 30 mm behind GT CAD in this sample.

Filter impact: original real geometry pixels 60,735 → 58,475, retaining 96.2789%. Target XYZ/depth arrays remain bitwise identical; only eligibility changes. Maximum retained sensor/CAD depth gap is 29.3822 mm. These are label statistics, not an improvement in model reconstruction or pose.

15 targeted tests passed on the existing remote environment, including actual CUDA BF16 invariance, exact-target gradients, unknown-label behavior, no CAD substitution, boundary handling, clean-input visibility, paired input invariance and fast/slow teacher parity.

Remaining limits:

- Some retained cases have roughly 10–14 mm mean sensor/CAD disagreement. This prevents treating both target definitions as exact physical ground truth. Native annotation equality proves provenance, not physical calibration accuracy.
- Full-raster CAD XYZ/camera checks contain sparse edge outliers up to 1.083 mm; the supervised mixed targets are internally consistent to the bound above. Initial stricter edge assertions failed and were split into interior and whole-raster diagnostics; no edge outlier was silently claimed absent.
- The analytic test certifies geometry projection, not texture/illumination fidelity or the factual correctness of the hidden CAD proxy. DINO feature supervision is disabled in the current geometry-only stage.
- These findings can corrupt supervision but do not establish that they explain the entire model-versus-LIP gap. No post-fix model gain is claimed.

Files: `summary.json`, `precision_before.json`, `precision_after.json`, `case00.png`–`case15.png`, and both full raw teacher audit directories. The synchronized pre/post audit archive SHA256 is `c1cec1bb7f9a0fdede8d3c17ac9ad45a4f9a1a11b66c304c620b2dc3ffaed75c`.
