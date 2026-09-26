# V34: accurate geometry first, 100-update paired pilot

The user's updated objective is accurate restored geometry with short,
evidence-driven iterations. This stage freezes the pose readout and removes
all DINO/pose objectives. No claim of accurate geometry or pose improvement is
made before the paired measurements. The default model is unchanged.

Both arms start from V33 full-window step2200, SHA256
`5cc8d8b774e8232b3a16a4678345819797627e4bf606c615550c0a51442d861e`.
All existing model tensors, including encoder and DPT, transfer exactly.
The optimizer resets because the objective and trainable parameter set change.
Complete optimizer/scheduler/all-rank RNG resume is verified at steps2 and3.

The control trains the current JEPA/DPT directly. The candidate predicts dense
pixel-to-reference flow from the SAME JEPA/DPT features and reads canonical XYZ
and camera depth from the estimated-pose CAD raster. Learned bounded residuals
(XYZ0.05diameter, depth0.25diameter) accommodate sensor differences and true depth
correction. A supervised gate mixes with the existing DPT fallback for surfaces
not represented in the current render. CAD dropout always disables the lookup.
This is a deterministic geometry lookup, not another encoder or pose estimate.
The transport is applied to the final decoder; coarse staged-RoPE remains the
existing trained DPT with direct geometry supervision.

Training targets project the teacher canonical point through the ESTIMATED
pose into the reference raster. Eligibility additionally checks reference
coverage and canonical surface identity within0.03diameter, so projecting a
back surface onto a front surface does not create a false match. No symmetry
orbit substitutes for the annotated textured-surface identity. GT coordinates,
eligibility, poses, or masks never enter the student decoder.

Natural-hidden regions retain CAD XYZ/depth targets. Artificially hidden
original-visible regions retain original measured XYZ/depth. Visible measured
points receive canonical correspondence supervision, not CAD depth replacement.
The loss balances real/proxy independently, includes full camera consistency,
small normal and validity terms, and supervises the coarse decoder. Candidate
flow/support supervision is explicit; its predicted gate cannot mask losses.
The pose head, DINO feature heads, and history modules are frozen. The online
encoder remains trainable and EMA is maintained without a teacher-DINO forward.

Each arm: seed42,8H20,batch32,100 updates. Uniform single frames cover full training
streams. Each frame has a current-frame empirical initializer error or a GT-noisy
training pose; no stale earlier-frame initialization is used. Actual RGB-D
hand/other-object cutouts provide natural/light/heavy training cases. Learning
rates are DPT/transport1e-4, other representation3e-5, encoder1e-6,50-step warmup,
cosine floor0.5. No package changes, additional seed, or official-test access.

Before/after geometry is measured on64 fixed records from physical sequences
excluded from training by the existing hashmod5 rule WITHIN the training split.
All arms use identical frames, estimated poses, crops, donors and targets.
Report full eligible regions, real/proxy separately, mean/median/p90 XYZ,
depth, XYZ/depth consistency,10mm accuracy, normals and validity. Report
oracle-reference coverage/lookup accuracy only as teacher diagnostic ceilings,
never as deployed results or whole-region performance. Save dense heavy-case
arrays to inspect spatial errors.

Pilot gate: at least5% lower heavy XYZ for both real and proxy versus the parent
AND same-budget control, with no more than5% depth/consistency regression.
No automatic continuation after100 updates. Passing this short gate only earns
broader verification; it does not complete the user's accuracy goal.

22 targeted geometry/DPT/serial/AMP tests passed before launch. Runtime is a
pinned source copy at `/tmp/dexycb_geometry_transport_v34`; outputs are under
`/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_transport_v34`.
