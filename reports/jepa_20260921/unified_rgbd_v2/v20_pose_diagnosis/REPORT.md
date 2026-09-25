# V20 versus old LIP: frozen failure diagnosis

No production model training or optimizer updates. Completed fixed40 physical-sequence diagnostics on119 selected frames spanning20 objects. Non-symmetric rotation comparison uses28 physical sequences and498 paired perturbation cases. Native comparison uses all23,200 val frames. Official test untouched.

## Main findings

1. **Rotation correction response is the primary observed failure.** A separate same-base component swap on119 selected natural frames gives25.21% ADD-S@0.05d with V20,41.18% using the old LIP rotation update plus V20 translation, and30.25% using V20 rotation plus old LIP translation (both old LIP components:46.22%). These are frozen conditional diagnostics, not a new hybrid model or full native score. At a10-degree error, old LIP without history reaches6.818deg; V20 remains10.026deg. V20 signed response gain is0.76%/0.39%/0.62% of ideal across X/Y/Z; LIP without history53.52%/53.35%/56.54%. The paired physical-sequence95% interval for the error gap is2.622 to3.802deg. Translation response exists but is weaker. Old LIP history is not required for this rotation advantage.
2. **Restored geometry has negligible influence on the current pose readout.** Turning completion off changes output by0.00649deg and0.0423mm on average; replacing predicted surface values and validity with GT changes it by0.00619deg and0.0378mm. These are paired output changes, not pose improvements. Oracle supplies geometry only at the readout, not JEPA's earlier staged RoPE pass. All weights stayed frozen. The learned relation gain is0.04952; mean completion weight across all crop patches is0.01310 (includes background, not an object-coverage statistic).
3. **A different final linear readout alone is not supported as the fix.** Five outer folds split by physical sequence, with regularization chosen inside each training fold, produce residual rotation errors: V20 fused9.897deg, patch9.945deg, object query9.927deg; LIP query6.125deg. Model weights never change. This is an exploratory fitted diagnostic within val, not independent downstream evaluation. It bounds simple readout capability; it does not prove that no information exists or that a nonlinear readout cannot help.
4. **The problem predates removal of DINO supervision.** At step40400 (before V20), the same rotation probe gives10.01499deg, versus10.02577deg at45400. Both have under1% signed correction gain. The rotation rows of the head changed during training (`head_weight_audit.json`); this is not an accidentally frozen rotation head.
5. **Training/inference trajectory mismatch amplifies the failure.** `frame_batch_training.py` uses the frozen crop-reference output to build all subsequent student inputs; student outputs never update those training bases. Old LIP's `stream_training.py` updates `accepted` from its own predicted pose.32 actual train episodes replayed with reference versus student feedback give post-anchor median canonical rotation errors19.82deg versus37.13deg and median center error0.0637d versus0.1013d. This replay retains synthetic occlusion, and differing crops alter its realization; it is supporting distribution evidence, not an isolated proof of causality or a symmetry-reduced score.

## What is and is not entering the latent

Matched old/new crops are exactly equal. Changing the estimated pose does change V20 representations: the mean relative L2 change for +/-10deg is10.18% before JEPA,12.76% at patch output,20.88% at object query. The corresponding LIP query change is73.14%. Therefore the geometric input is not completely disconnected; variation is not organized into a reliably readable rotation correction.

The existing recovered-relation vector contains canonical XYZ, nearest canonical CAD residual, relative depth, matched depth residual, and `depth - (R_base X)_z`. It has no explicit same-pixel camera-XYZ or reprojection-XY residual. In the optical-axis example, a10-degree rotation produces exactly zero Z-consistency error but5.847mm XY error for a150mm object. Other network paths can still encode in-plane rotation; this is a concrete missing geometric cue, not a theorem that the entire model cannot estimate it.

A suitable explicit relation to test within the existing unified readout is:

`r_camera = R_base (diameter * X_recovered) + t_base - depth_recovered * K_crop^-1 [u,v,1]`

Use observed depth on trusted visible pixels and confidence-weighted predictions only where needed. Preserve source/confidence distinctions.

## Training protocol gap

Current `pose_pair_frames` is empty. Joint unfreezing restored pose gradients but retained the JEPA-only fixed-reference training layout; it did not restore same-observation/different-estimate paired correction and correct-pose zero-update training. Old LIP's saved config selects GT-centered/noisy starts in the retained trainer, which feeds back its own rollout; this JEPA factory starts from native PoseCNN and optional extra noise. The new training replay is not dominated by tiny errors: reference rotation median16.97deg, and34.13% exceed30deg. Do not explain this run as only having easy near-correct training examples.

## Repair priority supported by these results

First train the original pose head to respond correctly to signed pose perturbations and preserve a correct pose; require measurable rotation gain on held-out physical sequences. Add explicit camera/projection relations to the shared patch/object-query path if readability remains weak. Then introduce student-feedback trajectories gradually. Keep this inside the unified JEPA path, with no FP bypass. Additional DINO/DPT reconstruction training or learning-rate changes alone have not been shown to address the measured failure.

## Limits and audit

- Pure-LIP native baseline SHA89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868; V20 terminal SHAb74ccb3d5de69386fb440aec08f4a8fa9cd00781e63bd16759f16cab5694f29c. Split/mesh/initializers match.
- Native ADD-S@0.05d: LIP83.6642/53.4683/29.2782%, V2031.3342/10.1969/2.6433% for all/<50%/<30% visibility. Native LIP has history; controlled probes include no-history LIP.
- Old LIP's replay is bitwise identical to stored poses (max difference0). V20 eager diagnostic versus compiled native max pose-component difference0.0003993; therefore small threshold-level diagnostic changes should not be overinterpreted.
- GT bases, GT surfaces/validity and `patch_minus_gt_ORACLE` use unavailable GT information and are not deployable metrics. Predicted-geometry rigid fits also use GT support and include unsupervised visible regions; they are not a standalone geometry-quality verdict.
- Full model-state hashes match before/after every val probe. The train replay uses no optimizer and no backward. Ridge readout fits are isolated diagnostic models and never replace production parameters.
- These experiments establish functional bottlenecks and support repair priorities; they do not establish that any proposed repair will recover the52.33pp native gap.
