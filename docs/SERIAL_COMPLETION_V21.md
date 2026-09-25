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
