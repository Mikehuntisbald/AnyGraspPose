# V21: CAD-assisted completion consumed by pose

The user goal is to restore occluded appearance and geometry with CAD and use
that restoration for pose. V20 did not enforce this: pose read the shared JEPA
patch directly, with a weak recovered-geometry increment. V21 removes that skip.

```
RGB-D + estimated-pose textured CAD + cached CAD surface descriptors
  -> existing DINO4/11, pair CNNs, cross attention and JEPA
  -> decoded DINO4/11 appearance + canonical XYZ/depth/validity
  -> measured visible appearance/depth retained; predicted missing content
  -> full camera XYZ residual + signed geometric moments + local appearance
  -> object query -> learned pose increment
```

There is no FP/LIP feature branch, PnP, SVD solver, or raw-patch readout. Texture
is represented by decoded local DINO features, not a synthesized RGB image.
Both decoded feature tensors actually used by pose are the tensors exposed for
reconstruction. Pose gradients reach them and the final XYZ/depth predictions.

Measured camera depth is taken from the unclipped sensor crop, not the clipped
geometry encoder channel. Visible selection uses predicted visibility, never
GT masks. Missing depth does not create a measured point. Missing-region weight
is bounded at 0.5 and combines predicted support/validity. These weights are
heuristics, not calibrated uncertainty. Predicted canonical surface identity is
used even for measured depth: inverse-transforming the measurement by the base
pose would bake the pose error into the supposed correspondence.

Local residuals include all three camera axes. Cross products/covariances are
formed before patch pooling. A global moment token explicitly exposes signed
rotation relations to object attention. The learned update scales with the
weighted correspondence residual, so an exactly aligned complete surface cannot
produce a large appearance-driven update. No evidence gives zero update.

## Staged validation

1. A train-split-only cache has 192 training and 64 development records, separated
   by physical sequence (SHA256 modulo 5). Ideal GT-rendered XYZ/depth/features
   validate the readout in isolation. These are oracle inputs and not deployable
   pose scores. The first 600-step attempt failed the 1-degree zero-update gate.
   The residual-scaled revision passed: 10-degree perturbations became 3.17
   degrees; zero-base updates averaged 0.0051 degrees. Gate thresholds were not
   relaxed. This development holdout informed the revision; native val is separate.
2. A real RGB-D backward verifies pose-only gradients to decoded XYZ, depth,
   both appearance feature layers and shared patch. Complete resume is tested.
3. Only after the oracle gate, initialize the shared backbone from V20 45400 and
   the new pose readout from this oracle training. Reset Adam explicitly because
   the readout and objective changed. Run one seed42 stage capped at 1000 updates.
   Do not claim equivalence between these updates and previous 40-frame episodes.
4. Joint training uses predicted completion exclusively. Same observed frame is
   paired with different estimated poses; regular exact-GT bases train zero
   updates. A third forward uses the student's detached pose on the next frame.
   GT-based starts are training-only; native inference retains PoseCNN starts.
   A review found the first implementation generated cutouts independently in
   each estimated-pose crop. At complete checkpoint350, training was explicitly
   adapted to reuse exactly the same RGB-D crop and cutout for both estimates,
   re-rendering only CAD and updating the estimated-pose state. Every pair now
   asserts identical RGB and raw depth. Model/Adam/scheduler/all-rank RNG were
   restored exactly; the earlier source and checkpoint remain archived.
5. Native validation uses own feedback, no GT access, 23,200 frames. Compare to
   old pure LIP (83.66% ADD-S@0.05d) and V20 (31.33%), keeping conditional/oracle
   scores separate. Never replace the default model based on the oracle gate.

## Supervision boundaries

Artificially hidden originally-visible regions retain original real feature
and depth targets. Naturally hidden regions use GT CAD proxy targets. Unmasked
visible RGB/depth receive no completion loss. Their measured 3D points do receive
canonical CAD correspondence supervision for pose. GT transforms/renders are
confined to target construction; they are not fields of student observations.

The joint objective retains pose, dense missing geometry, CAD correspondence,
coarse geometry and weak DINO4/11/local-difference supervision. It adds direct
pose-response, paired-response and visible canonical-correspondence terms.
History remains off. The static CAD cache, source experiments and environment
are preserved. Source runs execute from isolated local-disk snapshots; artifacts
and full checkpoints are stored under the durable remote experiment directory.

## Step700 adaptation: retain the geometry readout

Native step500 improved ADD-S@0.05d from V20's31.33% to49.48%, and visibility<50%
from10.20% to27.42%. Controlled rotation remained weak:10 degrees became9.82.
On paired positive-axis cases, disabling completion worsened9.45 to12.27 degrees.
A separate CPU FP32 replay of the same oracle development cache found the
readout had regressed to8.00 degrees, versus3.17 at oracle initialization.

At complete checkpoint700, retain all parameters, Adam, schedule and rank RNG;
add weight0.25 readout-only oracle rehearsal on the **training** cache partition.
It supplies ideal canonical/camera geometry and cached student appearance,
never target features or GT into the JEPA/encoder forward. Only pose-readout
parameters receive this auxiliary gradient. The normal three training forwards
still consume predicted completion exclusively. Exact per-pixel visible masks
replace the previous patch-majority mask for observed CAD correspondence loss.
The stage endpoint remains1000; no automatic budget extension.

The legacy checkpoint provenance field `teacher_input:false` refers to the main
student JEPA forward. It does **not** describe this explicit readout-only oracle
training branch. The config's `oracle_rehearsal_weight` and adaptation receipt
record that branch; GT-free native inference has neither teacher nor rehearsal.
