# V20 pose plus simplified geometry and CAD correspondence

Source: joint V19 step40400, SHA256 `39f5b85cbf39541e0d25627b13ce0edb60f857113d91459be9f8c3a9bfdb3b60`. Continue5000 remaining updates to the existing45400 endpoint. Seed42,8H20,effective batch32. History off. Previous source/runtime remains intact.

Train the DINO input encoder, JEPA, DPT, CAD interaction, geometry relation readout, object query and pose head. Remove every DINO feature target from training: absolute feature distance/cosine, feature-error prediction, feature centering/global correspondence, multiscale local differences and local feature correspondence. Remove normal supervision. Freeze feature_mid/feature_last/log_error prediction heads at inherited weights; retain their dormant optimizer states and outputs for explicit diagnostic evaluation. No new network path.

Objective: original pose loss1.0; final XYZ0.25 and depth0.25; camera consistency0.1; geometry validity0.1; visibility/support0.1; direct CAD surface correspondence0.1. Real/CAD geometry factor1/0.5 is independent of removed CAD feature-loss weight. Retain0.1 times the weighted coarse depth/validity loss for detached RoPE routing; no coarse XYZ/camera duplicate loss. All geometry and confidence labels retain the prior real-hidden/CAD-hidden masks; visible unmasked content has no reconstruction loss.

Geometry-only teacher uses the same GT geometry rendering and masks. DINO teacher forward and hybrid RGB construction are skipped. EMA weights/counter remain a cheap shadow of the online encoder, not a training feature target. The frozen causal crop model and static CAD caches are unchanged.

All model tensors, Adam states, RNG and sampler are inherited. The global LR schedule remains35400..45400; no new warmup or LR jump at40400. Native source40400 evaluation and native/fixed40 diagnostics at41400/45400. DINO reconstruction metrics during frozen evaluation are diagnostic only and do not select this pose-focused model.

Source executes from local disk `/tmp/dexycb_pose_geometry_v20_local`; durable artifacts/checkpoints in `/mnt/why/dexycb_lip/unified_jepa_20260921/pose_geometry_v20`. Full checkpoint every50 updates. No automatic budget extension, official test, multi-seed or default model promotion.

Validation before launch:8 CPU tests passed (feature-output independence, no DINO-target access/gradients, pose/geometry gradient flow, compiled loss, optimizer/state regressions). H20 full40-frame forward/backward passed: geometry-only target tensors match feature-teacher geometry exactly (including NaN label masks); forbidden-forward hook confirms zero training teacher-DINO calls; frozen feature heads have no gradients. Existing Adam state inherited exactly. Pose-only gradients still reach patch, recovered XYZ and depth. Warm preflight2.518s, peak9.69GB; not an8GPU step benchmark. One preflight update discarded; source checkpoint untouched.

Eight-rank startup/resume audit passed at40403: initial model and entire optimizer exact, pose/JEPA updated, unused feature heads unchanged, history unchanged, EMA counter continuous. Controller advanced to native source40400 evaluation, followed by training/evaluation at41400 and45400.
