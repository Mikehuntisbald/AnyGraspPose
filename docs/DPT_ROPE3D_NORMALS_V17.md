# V17: DPT geometry decoding, CAD 3D RoPE, derived surface normals

The experiment replaces the independent-patch MLP geometry decoder with a DPT-style dense decoder. It continues JEPA recovery training only; pose/query/relation weights remain frozen and history stays disabled.

## Architecture

- Read the four **JEPA block outputs**, each 256 patches by 256 channels. Reassemble them at 64/32/16/8 spatial resolution, width64; refine from coarse to fine with spatial residual convolutions. Decode to224x224, five channels: centered object XYZ/diameter, signed camera-depth residual/diameter, and a validity logit.
- No encoder/RGB skip bypasses JEPA. The DINO feature recovery heads still consume the final patch latent. The DPT geometry output enters the existing measured/completed relation readout. The frozen pose output is not an accuracy claim during recovery-only training.
- The pyramid/reassembly and residual fusion follow the [DPT design](https://arxiv.org/abs/2103.13413), with project-specific width, output channels and deterministic separable bilinear operators. This is a newly initialized decoder, not pretrained DPT depth weights. The old MLP is removed from this model.
- Enable the existing [CAD 3D RoPE](CAD_ROPE3D_CANDIDATE.md): measured observation XYZ and cached CAD XYZ in common camera axes, four zero-initialized per-head gates, visibility/depth fallback. CAD Utonia features remain cached. No GT geometry is used as a RoPE input.

## Surface orientation supervision

Derive normals directly from predicted XYZ: normalize the cross product of horizontal and vertical finite differences, at1/2-pixel spacings. Supervise the oriented normal with1-dot(predicted,target), not absolute cosine. No separate normal head is introduced.

The normal loss is `0.1 * (L_real + 0.5 * L_CAD_proxy)`, with equal scale mass and normalization over eligible samples. All pixels along each stencil must lie in the same real/proxy supervision region; visible unmasked predictions receive no normal supervision. Reject invalid target geometry, degenerate tangents, near-collinear target tangents, and neighboring target points separated by more than0.05*spacing object diameters. Eligibility is independent of predicted validity/confidence. A collapsed prediction still incurs orientation error. Real-depth normal targets can remain noisy even after these checks.

Existing XYZ, depth, camera-consistency, feature, multiscale local-difference and correspondence losses remain. Normal supervision directly trains XYZ; depth is coupled through the existing XYZ/depth consistency objective. Teacher geometry is used only to construct losses and evaluation targets.

## Initialization and bounded training

Parent: the complete V16 checkpoint at23400, SHA256 `3f708470c890e98760f9d2a3107bd318f88eef860bd12af74bd8d741d0995e67`. All8 ranks stopped at that completed update. Shared tensors, online DINO, EMA weights/counter and sampler/RNG are inherited exactly. Only the old surface MLP is removed; DPT is random and RoPE gates start at0. AdamW and scheduler restart for this structural/objective change.

User budget:1000 additional updates,23400→24400, seed42, effective batch32,8H20,40-frame episodes, frame batch8. Peak learning rates: new DPT/CAD/RoPE1e-4, shared JEPA1e-5, online DINO1e-6;50-step warmup followed by cosine decay to0.1 of peak. Checkpoints every50 updates. V16 execution optimizations remain enabled.

Two updates followed by a new-process one-update resume exercise the full checkpoint path before the remainder. Fixed40 recovery evaluation at+500/+1000 and a paired source-MLP evaluation use the same immutable step11000 EMA feature teacher. Report real/CAD XYZ and depth errors, normal angular errors and valid stencil counts, spatial retrieval, CAD correspondence, and normal/error visualizations. Three changes are combined; this run does not isolate the individual benefit of DPT, RoPE or normal supervision. No automatic budget extension or default model promotion.

CPU checks cover the decoder's cross-patch gradients and all four inputs, deterministic resize parity, narrow checkpoint migration, optimizer groups, fullgraph forward/backward, RoPE fallback, normal orientation/sign, invalid boundaries, source isolation and no teacher gradients. H20 preflight and startup audit receipts are recorded separately with the run.
