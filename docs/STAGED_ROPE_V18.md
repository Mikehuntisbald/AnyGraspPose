# V18: measured-first, recovered-later CAD RoPE

The student remains one four-block JEPA backbone with the V17 DPT decoder. This stage changes which geometry positions its CAD attention, retains every trained tensor and Adam moment, and adds1000 updates from24400 to25400. Pose/query/relation weights remain frozen; history stays disabled.

## Forward path

1. Before block2, use measured surface positions only where predicted target visibility>=0.7, valid-depth mass>=0.5 and normalized depth standard deviation<=0.05. Depth statistics come from the actual student depth, including artificial occlusion. Invalid or mixed patches fall back to the existing content/2D attention.
2. Complete the four-block coarse JEPA pass with measured-only CAD positions, then run the **same existing DPT weights** on taps1/2/3/4. This preserves the trained decoder input distribution. No second independent head is created.
3. Reliable measured positions take priority. Elsewhere, if the intermediate validity mass>=0.5 and predicted object support>=0.7 and predicted camera depth is finite and positive, backproject that depth through the actual student crop camera rays. Use the resulting surface as the late RoPE query position. Confidence is0.5 times validity mass times predicted object support, capped at0.5; RGB visibility alone cannot suppress recovery when actual sensor depth is missing. Unknown positions retain the ordinary attention path.
4. Recompute the shared block4 from its original pre-CAD-read input, using the new measured/recovered CAD positions. The final DPT pass uses taps1/2/3 plus the recomputed block4 and produces the existing XYZ, signed depth residual and validity outputs. There are four unique JEPA blocks, with block4 evaluated twice; coarse geometry never depends on the subsequent refined output. Final feature recovery still reads the shared final patch latent.

The thresholds are heuristic routing criteria, not an assertion that predicted validity calibrates geometric accuracy. Measured and recovered selections are disjoint; measured data are never averaged with a recovered hypothesis. Routing statistics exclude unavailable/dropped CAD references and distinguish measured, recovered and fallback eligibility.

## Coordinates and gradients

Both query and CAD key positions use current camera axes, a common origin at the estimated object center, and object-diameter units. Measured query positions are `(X_sensor_camera-t_base)/d`. Recovered query positions are `(ray*z_predicted-t_base)/d`; `z_predicted=t_base_z+d*predicted_depth_residual`. CAD keys are `R_base*X_CAD/d` for metric CAD coordinates.

The predicted canonical XYZ is **not** transformed through the base pose and substituted as camera position: that would re-impose the pose hypothesis. Predictions, confidence and routing coordinates are detached inside the positional branch. The intermediate decoder is trained through an explicit geometry loss, while the final JEPA still receives content/attention gradients. No GT pose, teacher features, masks or geometry enter the routing inputs.

The intermediate auxiliary objective is0.25 times the existing-weight XYZ/depth/validity/camera-consistency terms, using real-hidden and CAD-proxy masks separately (real factor1, CAD factor0.5). Original final reconstruction, local correspondence and normal losses remain unchanged. Non-occluded visible predictions do not gain a completion loss. Because routing and intermediate supervision change together, a whole-model comparison does not isolate their individual effects.

## Continuation and validation

Parent step24400 SHA256: `ec0732f609fd04dc7aee50a42c1ae7b9830a9e9acf8f1ac9a614efc45e50add5`. No new parameters or DPT reset. Preserve the complete model, EMA counter14650, Adam moments/steps/groups, sampler and8rank RNG. Restart the LR schedule at the same0.1-of-peak boundary, rewarm for50 steps, then cosine decay through25400. Seed42, effective batch32, microbatch4,8H20,40-frame episodes, frame batch8; save every50 steps.

Run CPU routing/coordinate/gradient/compile checks, real H20 forward/backward, then24402→24403 cross-process resume plus a full-state audit before continuing. Evaluate fixed40 at+500/+1000 against the pinned V17 terminal evaluation with the same fixed step11000 EMA teacher and crop protocol. Report coarse and final geometry, routing coverage on missing real/CAD regions, normal errors and CAD correspondence. At the end, also run the same-checkpoint RoPE switch probe. No automatic budget extension, official test or default-model promotion.

On three inspected training frames, the current complete-pass DPT validity maxima were below0.60, while object-support maxima exceeded0.98. Therefore recovery uses the conjunction of predicted support>=0.7 and validity>=0.5, and their product downweights the positional hypothesis. A preliminary0.7-validity-only gate rejected every recovery candidate; that preflight made no optimizer update. The complete coarse pass retains the learned DPT and support-head input distributions.
