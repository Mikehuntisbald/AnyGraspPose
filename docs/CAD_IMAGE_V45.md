# V45: explicit CAD-to-image correspondence through JEPA

Status: V45 completed100 updates and frozen64-record controls. Geometry/endpoint errors improved against its V44 parent, but robust pose did not improve. No promotion. V46 continues500 updates with corrected class normalization; its result is pending.

## Motivation and reference

The V44 dense map already associates each image pixel with a CAD coordinate.
It does not explicitly return where a specified CAD point occurs in the image.
Accurate point membership alone cannot establish correct registration.

[GoTrack, section3.1](https://arxiv.org/html/2506.07155v1#S3.SS1) predicts template-to-image flow and visibility, then lifts template points to3D and estimates pose with PnP-RANSAC and inlier refinement. Its two-branch transformer/DPT and training recipe are not reproduced here. This independent implementation takes the explicit endpoint/visibility and geometric-verification ideas, retains the unified JEPA, and additionally trains occluded front-surface correspondences for completion. Back surfaces do not receive image endpoint supervision. History remains disabled.

## Model

The shared DINO4/11, geometry CNN, cross-attention, four JEPA blocks, DPT, internal256 CAD tokens and full8192-point bank remain. A new reverse readout reuses **the identical DPT query and Utonia/dynamic-geometry key embeddings** already used by V44 image-to-CAD reconstruction. There is no extra encoder or FP/pose bypass.

For512 fixed, evenly indexed points from the8192 bank, similarity is normalized over a112x112 image grid. A local3x3 soft argmax around the global maximum returns a subpixel endpoint; global heatmap CE trains incorrect distant matches. Unlike dense XYZ retrieval, this spatial matching score has no detached coordinate-prior term. A small MLP predicts self-surface support and current visibility using the matched query, key and distribution diagnostics.

The output explicitly includes CAD point IDs, canonical3D coordinates, crop2D coordinates, support/visibility and confidence. Probe artifacts also invert the crop affine to give original-image coordinates. A robust PnP diagnostic consumes these prediction-only3D/2D pairs. Existing dense image-to-CAD pairs are independently solved by the same algorithm as a control. The old learned object query/pose head remains frozen and separately scored; PnP has not silently replaced the public model interface.

## Supervision and checks

Point targets are GT projections through the **existing student crop camera**. Labels are built after the student inputs. A GT z-buffer check within3mm, canonical-coordinate agreement within0.02d and silhouette interior exclude self-occluded/back surfaces and boundaries. Visible, artificially hidden originally visible, and naturally hidden CAD-proxy endpoints are normalized separately (weights1/1/0.5); current visibility is a separate label. The original real sensor depth targets are unchanged. Projection labels do not use the estimated pose as the answer.

On the first training batch, each rank independently checks FP64 native-camera projection followed by the crop affine against the teacher's endpoint labels (maximum allowed0.001px), and verifies that the endpoint objective alone supplies nonzero gradients to the shared patch latent. Unit tests cover pixel centers, diameter units, front/back and artificial occlusion labels, no-CAD handling, shared query/key gradients and PnP recovery with25% outliers.

Original geometry evaluation masks remain unchanged. Correspondence evaluation records source-specific pixel EPE,3px accuracy, visibility/support precision/recall, solver acceptance and pose errors. Pose metrics are controlled fixed64 physical-holdout probes with10-degree initial perturbations, not native tracking or official-test results.

## Reproduction

Config: `configs/jepa/cad_image_v45.yaml`. Parent: V44 supervised100, SHA256 `f651d6df004c7b025a7a827785a29ff1dd48621add96d5aad5aa8c65beabd6a9`. Parent tensors transfer exactly; only the point-visibility MLP is new. Fresh optimizer, seed42, eight H20s, effective32 observations/64 hypotheses, strict2-to100 resume, every50-step checkpoint. No environment changes, multi-seed or official test.

Runtime: `/tmp/dexycb_cad_image_v45`. Artifacts: `/mnt/why/dexycb_lip/unified_jepa_20260921/cad_image_v45`. The pinned runtime must not be edited during execution.

Remaining limitation: endpoint regression and a true CAD point bank do not themselves ensure reliable matches under severe occlusion; full native validation is required before adoption. The first solver diagnostic uses finite positive confidence and geometric consensus; calibrated visibility rejection needs a separately labeled frozen comparison.

## Completed V45 result and V46 correction

Equal physical-sequence means on the heavy fixed64 subset: real CAD XYZ14.422→13.376mm; proxy XYZ18.336→15.561mm. Real depth12.505→12.422mm; proxy depth13.360→12.826mm. Correspondence EPE observed11.196→8.402px, artificial-hidden11.408→9.232px, natural-hidden11.531→8.242px. These gains still do not beat original V38 real XYZ9.482mm and proxy14.996mm.

Raw forward PnP rotation worsened13.624→14.593deg and translation61.781→72.576mm; the base has10deg/0mm by probe construction. Frozen visibility gating after training rejected every heavy update. Support recall was10.1%, current-visible recall0.94%; returning the base for every frame is not a refinement gain. Soft search priors at16/32px were also recorded, but visibility collapse makes their gated terminal results all-fallback, not evidence of successful localization.

V46 changes only supervision configuration: positive and negative support/visibility classes get equal mass within each example, and the classification coefficient increases0.1→1.0. No new parameters. The test proves adding100 negative copies cannot reduce a positive sample's gradient mass. Classification scores after rebalancing are not calibrated deployment probabilities. V46 starts from V45 terminal checkpoint d67501ab547c40fcdebc84a67813e825121188ef73d4eb297da5b66c7097efcf, retains geometry/endpoint targets, and runs500 more updates with a fresh optimizer and strict2→500 resume. Its pinned runtime is `/tmp/dexycb_cad_image_v46`.32 targeted tests pass.

The subsequent native40 diagnostic uses fixed previous-frame sealed LIP poses for crop/base, three fixed frames per sequence (duplicates removed), and natural/heavy inputs. It is conditional one-step evaluation, not closed-loop tracking. Its purpose is to avoid mistaking performance on10-degree/zero-translation probes for native pose quality.
