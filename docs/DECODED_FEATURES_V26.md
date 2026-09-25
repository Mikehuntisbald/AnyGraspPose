# V26: retain JEPA's fused features at visible as well as hidden patches

Status: implemented, tests and paired200-update experiment completed. The heavy
pose/reconstruction gate failed; no model promotion. AMP gradient fix from V25 is retained.

## Why this change follows the intended pipeline

The requested main flow is observation/CAD/geometry/history→JEPA→patch latent→
object query→pose, with restored content actually supporting pose and measured
depth retained as a constraint. The V21 packing code violated the intended
feature ownership at predicted-visible patches: it replaced decoded JEPA features
with the original DINO observation. In a same-crop paired-estimate example those
observation features are identical even when the estimated CAD view changes.
The fused JEPA representation was therefore absent from that appearance readout.
Earlier hidden-only gradient tests did not test this visible-patch path.

V26 uses the SAME decoded DINO4/11 tensors for pose at ALL valid patches. It does
not add a raw-patch pose branch, FP projection, or independent visual encoder.
Original RGB still enters the student; reliable measured depth still overrides
predicted depth exactly. Reconstructed geometry remains a confidence-weighted
hypothesis. Teacher source/domain rules are unchanged, including no feature/depth
reconstruction loss on originally visible, unmasked pixels.

This remains valid even if pose is restricted to decoded features rather than
raw JEPA patches; the optional architecture clarification is not needed for this
repair. The visible-JEPA feature substitution is tested independently from that
broader choice. Source mode is explicit in config; older checkpoints default to
their legacy behavior for faithful reproduction.

## Paired experiment

Use exact V21 step1000 model/Adam/eight-rank RNG with V25 AMP fix, same seed42,
learning-rate schedule, scalar losses, paired crops, feedback state and200 updates
to1200. The completed V25 run is the matched control. Only
`serial_completion.readout_feature_source: decoded` changes the readout input.
There are no new learned parameters and no pose-head reset. The original legacy
pose-head structure is used, not V24's frozen shape-conditioned head.

The controller exercises eight-GPU gradient preflight and1002→1003 strict resume,
then full native23,200-frame validation, fixed40 restoration and controlled pose.
It stops at1200 and does not expand budget. Improvement must be measured in native
pose and restoration; an active gradient alone cannot satisfy the original goal.

13 tests passed, including nonzero decoded-feature pose gradients on VISIBLE
patches, independence from directly supplied observation features at fixed decoded
features, exact measured-depth retention, legacy behavior, AMP preview gradients,
shape conditioning and optimizer migration.

Runtime `/tmp/dexycb_decoded_features_v26`; artifacts
`/mnt/why/dexycb_lip/unified_jepa_20260921/decoded_features_v26`.

## Completed result: architecture path corrected, checkpoint not adopted

Both training and all evaluations completed. Initial model, Adam, scheduler and
all-rank RNG hashes match V25 exactly; normalized config differs only in output
path and feature source. Native and recovery frame/teacher/crop/donor/pixel
identities were verified after evaluation.

|Route|All ADD-S@0.05d|Visibility<50%|Visibility<30%|
|---|---:|---:|---:|
|V25 observed-visible|50.520%|27.196%|7.968%|
|V26 all-decoded|50.825%|21.947%|7.719%|

All-frame changes+0.305pp, but heavy−5.249pp and extreme−0.248pp. This does NOT
pass the restoration/pose objective. The terminal weights are retained for audit,
not promoted; the budget is not expanded because a wiring test passed.

Heavy real XYZ/depth40.514/17.243mm (control39.936/17.740); proxy39.848/24.311mm
(control38.361/24.185). Canonical XYZ worsens. Controlled non-symmetric positive
10-degree rotation ends at9.144 degrees (control9.018), zero-base drift2.370
(control2.479). Restoration quality remains inadequate for reliable pose.
Terminal SHA256:159c50048f08ffaca4111080ccb83a80cfb4967c9227bbbf39741b52223fc977.

The two stacked256→384 feature projections have full column rank256, condition
number21.13 (parent23.22). This is a linear algebra result BEFORE AMP quantization,
normalization and source substitution; it does not prove accurate pose information
or that a finite learned readout can recover it. Do not blame a dimensional rank
collapse without evidence. The next bounded probe should read actual predicted
latents/geometry on independent sequences, rather than substituting ideal geometry
and treating oracle success as usable restoration. Any later raw-patch readout
choice must remain consistent with the user's main JEPA flow and restoration-use
requirement; no FP or independent encoder shortcut is authorized by this result.
