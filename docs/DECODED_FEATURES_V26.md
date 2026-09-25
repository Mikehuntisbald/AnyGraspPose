# V26: retain JEPA's fused features at visible as well as hidden patches

Status: implemented, tests passed, paired200-update experiment started. No model
promotion or accuracy claim yet. AMP gradient fix from V25 is retained.

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
