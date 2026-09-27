# V67: better oracle use, not yet reliable learned geometry

64 frozen cases, same seeds/regions/checkpoint as V66. All normal records match V66 exactly; all first rounds are invariant. Tests: 9 passed in 4.75s in the existing remote environment. No training or default promotion in V67.

Surface-preserving feedback uses 16 actual raster samples per patch and nonnegative validity-weighted canonical similarity, replacing mean XYZ and its mismatch penalty. It changes both aggregation and scoring, so gains cannot be attributed to pooling alone.

At 60-degree initial error and requested heavy occlusion, real-hidden flow EPE is 27.7115px normally, 26.7697px with predicted geometry at strength2, and 26.7759px at strength8. Oracle XYZ at strength8 gives 17.0418px; oracle XYZ+validity gives 12.6615px. The corresponding V66 oracle XYZ+validity was 27.4665px. Thus the new mechanism can use accurate recovered geometry more effectively, but its learned predictions do not currently realize that potential.

At 10-degree heavy real-hidden points, normal EPE is 5.8033px, surface2 is 5.8210px and surface8 is 6.0304px. Visible points also regress (5.1764 to 5.3982/6.2391px). This prevents promotion. All conditions including adverse/sparse proxy slices appear in REPORT.md.

Actual heavy real-hidden geometry is essentially unchanged: 10-degree canonical XYZ 13.4582mm normal versus 13.4658/13.4641mm; 60-degree 37.5776mm versus 37.5692/37.5737mm. Matching changes must not be presented as meaningful reconstruction improvement.

Next bounded experiment V68: fresh paired control/surface training from the same V60-extra weights, 200 updates each, seed42, no pose/history/DINO feature losses. Surface arm persists strength8 in config and adds direct geometry-score correspondence CE (0.05 inside flow loss, whose outer weight is0.2). Existing appearance CE, endpoint and geometry targets are unchanged. This addresses the missing global geometry-match supervision; it is a combined candidate, not an attribution-isolating single-factor ablation. Labels remain outside forward.
