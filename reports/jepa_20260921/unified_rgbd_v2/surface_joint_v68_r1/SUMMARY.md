# V68r1: matched training completed; geometry goal not achieved

Both arms completed200updates from identical V60-extra weights, same seed42, windows and learning rates;32tests passed. Complete-state resume at2steps succeeded for both arms. All8rank losses/gradients were finite. The4,800 recorded recovery-CE terms are0–7.4451, below the theoretical13.5452 bound; the invalid-key soft-target defect no longer appears. Checkpoint hashes were recomputed against actual remote files.

The corrected control and candidate both condition soft targets on valid crop keys. Candidate additionally uses surface-preserving geometry feedback, strength8, and direct geometry-score CE. Training/evaluation masks are unchanged. Evaluation compares32/64/32physical-holdout training-partition frames at0/10/60degrees; identical frame identities, region pixel counts and flow point counts were verified. No official test or native pose evaluation was performed.

| Heavy condition | Metric | Before training | Control200 | Candidate200 |
|---|---|---:|---:|---:|
| 10deg real-hidden | Canonical XYZ mm |10.798|11.255|11.255|
| 10deg real-hidden | Depth mm |8.749|9.726|9.730|
| 10deg CAD proxy | Canonical XYZ mm |11.330|11.139|11.368|
| 60deg real-hidden | Canonical XYZ mm |36.448|35.424|34.969|
| 60deg real-hidden | Depth mm |17.089|18.361|18.290|
| 60deg CAD proxy | Canonical XYZ mm |35.940|37.481|37.121|

At60deg/heavy, candidate improves real-hidden XYZ by0.455mm versus control (13/16frames improve), but this is only about1.3%. At10deg/heavy, real geometry is effectively unchanged versus control and worse than before training. Some nonheavy60deg flow improves substantially (real-hidden EPE26.647→20.397px), while the corresponding XYZ change is only55.544→55.058mm. Flow improvement is not sufficient evidence of accurate reconstruction.

Decision: no default promotion, no budget extension. Retain the supervision correction. Surface feedback is a useful experimental mechanism, but this training did not achieve accurate geometry. Next diagnose whether flow-aligned CAD identities can directly improve the atlas decoder's canonical search prior, rather than only entering through the weak latent write. That frozen intervention is V69; it must not be mistaken for a trained JEPA improvement or a depth improvement.

Artifacts: REPORT.md contains all conditions and paired win fractions; paired_geometry.png/PDF are standalone figures. training_audit.json records the training checks; checkpoint_verification.json records actual hashes/sizes. Remote final checkpoints are control/seed42/last.pt and surface/seed42/last.pt under unified_jepa_20260921/surface_joint_v68_r1 (474,007,140bytes each). Full checkpoints remain remote; compact logs, frame results, receipts and plots are local. Remote project root is not a Git checkout.
