# V40: paired estimates and visible correspondence first

V39 restored original RGB-D to the input, yet JEPA CAD-flow EPE did not improve.
This short curriculum tests whether correspondence can learn from available
surface evidence before expecting heavy-occlusion recovery. Clean-input success
is a prerequisite only; the full geometry-restoration goal is unchanged.

Parent is V38 raw200, SHA256
`e1e19e3a20e3d7e81e3090043a6d09cc8aee9db4156c2d1cee4dd709bfd3f9ce`.
Every existing model tensor transfers exactly. Both arms use the same fresh
optimizer, learning rates, seed42 and100-update schedule. The canonical surface
loss, real-depth targets, mandatory CAD lookup, no-free-XYZ-offset policy and
flow weight12.5 match the declared V38 candidate formulation in BOTH arms.

Each of32 distinct observations per update receives two pose hypotheses with
inverse camera-frame rotation errors and opposite center-translation errors.
The second hypothesis shares the exact first crop, RGB-D, intrinsics and
occluder. Only its reference rendering and estimated-pose state change. Rotation
noise std0.2rad and translation noise0.035diameter are training GT perturbations;
every tenth update includes exact-zero examples. This curriculum deliberately
uses bounded GT-noisy pairs, not potentially invalid mirrored native estimates.
No such GT perturbation/reset is introduced into deployment inference.

The corrupted arm receives the normal textured artificial occlusion. The clean
arm receives the original current sensor image/depth. Both retain the same
original artificial/natural target masks and the same visible-correspondence
loss domain, so restored evidence does not silently change target weighting.
The model processes64 pose hypotheses per update; checkpoint sampler position
counts32 distinct observations. No history, pose loss or DINO feature loss.

Encoder/JEPA/DPT/transport train jointly; pose and feature readouts stay frozen.
Both arms have8H20,microbatch4 distinct scenes,100 updates, checkpoint every50,
and exact2→100 resume. Before/after probes use the same64 physical-holdout records
under BOTH corrupted and clean current inputs, with scoring masks unchanged.
Report old raw XYZ, CAD-canonical XYZ, depth, CAD-flow EPE and identity baseline.
The clean control is not a heavy-occlusion deployment score or native accuracy.

13 unit/regression tests pass. Eight-rank preflight verifies identical RGB-D
between hypotheses and gradients into JEPA/DPT/transport. The runtime is pinned
at`/tmp/dexycb_visible_correspondence_v40`; artifacts are under
`/mnt/why/dexycb_lip/unified_jepa_20260921/visible_correspondence_v40`.
No automatic extension beyond100 updates or default-model promotion.
