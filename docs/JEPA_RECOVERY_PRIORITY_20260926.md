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

## V51–V52: shared recovery gradient balance

V51 measures32 distinct training observations/64 paired hypotheses without optimizer updates. The CAD correspondence gradient at the final shared patch is7.78–15.26 times the complete geometry gradient; at dense DPT features5.59–9.13 times. Depth/CE gradient cosine is mostly near zero (patch−0.083 to0.231). This establishes a local gradient imbalance, not a universal loss conflict or proven generalization cause. Final depth-output convolution receives geometry but no correspondence gradients; shared JEPA/DPT receives both.

V52 is a bounded matched control: source V44 supervised100, normal diverse sampling, seed42,200 updates per arm. Only correspondence weight differs (1.0 versus0.1); frozen pose/history and disabled DINO losses are unchanged. Two arms run sequentially on8 H20s, save every50, and verify full2→200 resume. The usual64 recovery frames are paired; another64-frame confirmation with seed offset54000000 is fixed before training. It is another sample from the same held-out sequence pool, not an unseen-object or new-sequence test. No promotion is automatic. Physical camera XYZ/depth and camera-surface orientation are reported separately from canonical CAD identity.

Status: V52 completed both200-update arms and both recovery probes. Architecture and learning rates unchanged. The usual heavy probe (control→balanced) gives real CAD XYZ13.298→12.278mm, depth12.558→10.424mm; proxy CAD XYZ15.300→16.566mm and depth12.796→13.026mm worsen slightly. On the prespecified confirmation64 (no stream/frame overlap with the usual64), real CAD XYZ12.769→11.604mm and depth13.283→11.859mm; proxy CAD XYZ13.050→11.725mm and depth11.353→9.653mm. Both heavy subsets span29 physical sequences from the same held-out training pool. This supports weighting as a contributing cause, not a uniform repair or accurate-recovery completion.

Both arms passed26 tests and strict2→200 resume;715 initial model tensors and all8 ranks' initial forward metrics match exactly.35 pose tensors remain bitwise unchanged. Source/terminal comparisons, physical camera XYZ, and camera-normal metrics are in the paired report. Normal-angle reductions do not establish an accurate surface. No pose experiment, default-model promotion, or automatic extension was launched.

[Full V52 evidence](../reports/jepa_20260921/unified_rgbd_v2/recovery_balance_v52/REPORT.md).

## V53: formal training requested

The user explicitly requested formal training. V53 continues the V52 balanced200 complete state for5000 new updates (global5200), preserving Adam/RNG/sampler and extending the LR schedule continuously. The network and supervision are unchanged.29 startup tests passed; the launcher schedules both recovery probes after new500/1000/2500/5000 updates. See [the active formal configuration and protocol](JEPA_RECOVERY_FORMAL_V53.md). This new authorization supersedes the previous bounded-comparison-only scope; it does not promote a default model or restore pose training.
