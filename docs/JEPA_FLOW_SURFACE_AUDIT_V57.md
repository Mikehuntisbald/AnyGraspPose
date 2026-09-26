# Frozen flow / geometry information audit

V57 uses V56 step200 without optimization. It answers whether accurate enough
surface evidence already exists in predicted template endpoints and recovered
depth before introducing a stronger recovery decoder.

`flow_surface_audit.py` implements three prediction-only geometric probes:

- Structured template triangles carry canonical coordinates along predicted
  endpoint flow. Folds and large canonical discontinuities are rejected. Holes
  explicitly fall back to V56. This affine triangle approximation is audited
  with a separate oracle-endpoint control.
- Existing PnP-RANSAC lifts predicted template correspondences into a rigid CAD
  rendering. No teacher value determines solver acceptance or candidate choice.
- Predicted endpoint rays multiplied by DPT depth supply metric3D points for the
  existing robust weighted rigid fit. The measurement variant combines actual
  depth with recovered depth using point visibility and depth disagreement, with
  measured contribution separately reported.

Oracle endpoints and direct GT CAD rendering are constructed only after the
prediction branches, solely for evaluation controls. Pixel scoring keeps original
real/proxy masks and counts fallback pixels. Coverage and solver acceptance are
reported separately. This is not a new model forward path or pose evaluation.

Final runtime: `/tmp/dexycb_flow_surface_v57_r4`.
Remote results: `/mnt/why/dexycb_lip/unified_jepa_20260921/flow_surface_audit_v57_r4`.
Source/model hashes, per-frame results, raw visibility scores/labels and separate
initial/intermediate audit versions are preserved.

The final report records partial metric-geometry improvement, severe depth errors
from RGB-only fitting, weak point-visibility discrimination and the observed
real-sensor/CAD target discrepancy. It does not promote any candidate or claim the
geometry-accuracy goal has been achieved.

`probe_teacher_pixels_v57.py` additionally compares cached poses with the native
NPZ annotation, centered and original CAD transforms, appearance/geometry mesh
bounds, and rendered XYZ/depth/ray consistency. The strict pixel/ray gate remains
reported as false on the fixed example; this diagnostic does not silently relax
its thresholds. `--require-strict` returns a nonzero exit when that gate fails.

A follow-up convention check consulted the official
[RealSense alignment implementation](https://github.com/realsenseai/librealsense/blob/master/src/proc/align.cpp),
which reprojects pixel locations while copying original depth samples. This is a
reason to verify the dataset's depth/color coordinate convention and calibration;
it is not evidence that the observed13.6mm discrepancy is caused by that convention.
No depth correction or target relabeling has been applied.
