# Active priority: JEPA recovery accuracy first

The user redirected the work away from pose/readout optimization. V48 was stopped through its launcher, retaining the full step150 checkpoint and the interruption receipt. This work has no queued pose/PnP follow-up. V45–V47 evidence remains preserved; those candidates were not promoted.

## What is trained and what is measured

Prioritize the JEPA representation, DPT geometry decoder and CAD interaction. Pose modules remain frozen, history remains disabled, and DINO feature loss remains off as previously requested. CAD coordinates, original image endpoints and depth are distinct outputs:

- `surface_xyz` is normalized **CAD/object-coordinate identity**, not camera-space XYZ.
- `surface_depth_m` is recovered camera depth. Physical camera-space XYZ follows from `crop_rays * surface_depth_m`.
- Explicit CAD-to-image endpoint readouts added in V45–V48 are archived experiments. They are not the current training objective.

Acceptance must use independent masked reconstruction, accurate local correspondence and surface detail. Training loss, semantic cosine, and pose/PnP scores cannot substitute for these. Originally visible then artificially hidden depth remains the original real sensor target; originally hidden regions use GT CAD depth/identity. Existing boundary/conflict exclusions and original evaluation masks are retained. Visibility does not turn predicted completion into a measured fact.

## V49: verify fitting ability before attributing failure to readout

The pre-readout V44 geometry checkpoint initializes a fixed32-observation/64-hypothesis training diagnostic, seed42/eight H20s,200 updates. No new architecture or pose branch is added. Encoder/JEPA/DPT/CAD projections train; old pose and feature heads remain frozen. Every update checks bitwise-identical observed inputs, base estimates, geometry labels and masks. Hashes across the2→200 resume match on every rank.26 targeted tests passed.

Training-only first/last-forward equal-rank means:

|Region|CAD identity XYZ mm|Depth mm|
|---|---:|---:|
|Artificially hidden real|15.766→6.041|11.851→3.780|
|Naturally hidden proxy|14.805→5.385|11.919→3.400|

The independent64 heavy probe gets worse: real XYZ14.422→23.678mm, real depth12.505→14.255mm; proxy XYZ18.336→27.864mm, proxy depth13.360→17.944mm. This deliberately small-set experiment demonstrates trainability and overfitting, not accurate recovery or a candidate for promotion. Its training and independent metrics also use different populations/mask policies; compare only paired changes within each.

Source SHA256 `f651d6df004c7b025a7a827785a29ff1dd48621add96d5aad5aa8c65beabd6a9`; terminal SHA256 `0e808877bd97fe830684eadaa88a4d88a093bdc965a4ceb778996bf0657ad055`. Full terminal checkpoint/source archive verified locally. Runtime `/tmp/dexycb_recovery_fit_v49`; artifact root `/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_fit_v49`.

## V50: exclude a simple recovered-position routing explanation

Same frozen V44 checkpoint and original64 masks; no training or pose evaluation. Only recovered RoPE trust or all RoPE gains are zeroed as explicit interventions.

|Routing|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|Original|14.422|12.505|18.336|13.360|
|Measured positions only|14.430|12.503|18.344|13.348|
|All RoPE off|14.427|12.503|18.351|13.349|

Every absolute change is below0.02mm. This rules out a useful immediate improvement from these frozen switches on this probe; it does not prove RoPE cannot affect training. The checkpoint remains unchanged.

Next work must target independent local correspondence and geometric reconstruction, preserving this evidence and supervision audit. Do not resume the cancelled V48 readout or use a pose-success gate to decide whether JEPA is accurate. The overall accurate-recovery objective remains unmet.
