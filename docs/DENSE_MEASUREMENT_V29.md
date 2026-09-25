# V28–V29: actual sensor ownership under occlusion

Status: selector trained and frozen-head intervention completed. The selector
improves sensor-point recall but has NOT improved pose; it is not enabled in any
default model. Original performance acceptance against pure LIP remains unmet.

## Why test this

The serial readout preserves actual depth only if an entire14×14 patch has
predicted visibility≥0.7. The visibility head reads pre-JEPA observed DINO
features. In V27's disjoint-sequence heldout population, added occlusion leaves
only12.97% of true valid object depth accepted. An oracle GT visible-fraction
patch selector still retains only40.73% (86.95% precision). This identifies a
granularity limitation, not the whole pose failure.

V28 trains equal-capacity LN/Linear visibility heads on frozen raw, fused, or
final-patch features for1000 updates. None meets both precision and recall;
simply selecting a different feature layer is not adopted. V28 uses the same
source/samples/split as V27, with soft BCE and a fixed0.7 threshold.

## V29 selector and safety boundary

A small pixel CNN reads normalized actual RGB, estimated-pose CAD RGB, the9
existing geometry channels, and the frozen static Utonia CAD mean descriptor.
It outputs only sensor-ownership confidence. It has no pose output, predicted
completion input, or trainable-DINO features. The CAD descriptor conditions32
channels; residual convolutions operate at224/112 resolution. CAD-off masks
the reference, geometry reference channels and descriptor before processing.

Training is1000 steps, seed42, AdamW1e-3, batch8, BCE+0.5 Dice, threshold0.7.
This jointly changes spatial architecture and supervision, not a single-variable
architecture ablation. Only the final1000-step weights are evaluated downstream.
750-step metrics exist but no750 checkpoint was selected or deployed.

Training curves use pixel bounds. `strict_metrics.json` separately recomputes
both selectors on the exact downstream eligible domain: full valid14×14 patch
AND finite positive measured depth. Earlier curves are preserved unchanged.

|Heldout subset|Legacy precision / recall|Dense precision / recall|
|---|---:|---:|
|All|93.29% /64.40%|92.63% /86.86%|
|Artificially occluded|77.66% /12.97%|78.25% /80.95%|
|Natural|93.87% /73.36%|95.45% /87.89%|

These are sensor-pixel metrics over the controlled training-data holdout,
not native visibility-bin pose accuracy. Four poses share each RGB-D image;
they are not four independent observations.
Here a positive means RGB-visible label AND valid sensor depth. It does not
prove geometric agreement with the CAD surface. V30 checks that separately.

## Frozen downstream intervention

The exact same V21 JEPA predictions and pose weights are reused. Only
measurement ownership/confidence changes. No optimizer step occurs, and the
legacy re-read delta is checked bitwise against the original forward in every
case. GT is used only for metrics and controlled estimate construction.

|Condition|Original rotation|Dense selection rotation|
|---|---:|---:|
|Non-symmetric ±10°|9.025°|9.126°|
|Non-symmetric added occlusion ±10°|9.341°|9.797°|
|Correct estimated pose|2.089° drift|2.717° drift|
|Mixed perturbation|11.857°|11.794°|

In the augmented rotation subset, true measured points increase from2.48% to
26.61% of total correspondence weight, but false measured points also increase
from0.81% to6.38%. A higher pixel recall alone cannot certify useful pose
correspondences. Reconstructed canonical XYZ and the learned readout still
need separate checks; a negative frozen-head intervention also includes input
distribution shift and does not prove a jointly trained selector is useless.

`pack_completion` now accepts an optional detached dense probability, disabled
by default. Exact actual depth is retained where selected; absent depth remains
absent; predicted completion receives at most0.5 confidence. The change does
not change appearance routing. It is an experimental interface, not a model
promotion. Ten targeted tests pass, including dense ownership, pose-gradient
isolation from the selector, missing depth, invalid crop bounds, CAD-off NaNs,
legacy behavior and decoded-feature gradient paths.

Remote artifacts: `unified_jepa_20260921/visibility_probe_v28`,
`dense_measurement_v29`, `dense_measurement_pose_v29`; each has an immutable
source archive and receipts. Local report copies include fit checkpoints and
hashes; large frozen caches remain remote. Remote runtime copies are not Git
checkouts and do not establish remote Git HEAD alignment.
