# Serial pose training: limited time coverage and late-frame degradation

This is a verified implementation/data audit, not a causal claim that sampling
alone explains the pose gap. The current V32 experiment is kept unchanged so
its shared-patch comparison remains interpretable.

`Factory.sample` reads a window beginning at a stored PoseCNN initializer.
`train_serial_completion.py` supervises relative frame0 or8, a second estimated
pose on that same frame, then feedback on frame1 or9. Additional decoded frames
in the12-frame episode never enter its loss. Initializer options are restricted
to absolute frames0–19; in the eligible physical training partition the latest
supervised absolute frame is28.

After applying the actual physical-sequence holdout exclusion:

-20,930 initializer options from4,813 streams;
-64,851 distinct frames can be reached by offsets0/1/8/9;
-350,018 total frames in those eligible streams;
-reachable fraction18.53%, before considering actual finite sampling frequency.

This describes the serial pose curriculum, not every earlier JEPA pretraining
stage. It does not mean only18.53% of objects or sequences have been trained.

The exact same23,200 native validation frame identities were then stratified
at absolute frame28, using the established object-macro ADD-S@0.05d metric:

|Region|Frames|Pure LIP|V26 JEPA|
|---|---:|---:|---:|
|All|23,200|83.664%|50.825%|
|Frames0–28|9,280|85.819%|75.151%|
|Frames29+|13,920|82.224%|34.622%|
|Early, visibility<50%|1,033|53.888%|35.291%|
|Late, visibility<50%|1,333|50.003%|14.857%|

Late-heavy contains19 objects; early-heavy contains20. Do not interpret the
macro difference between these bins as a controlled change in difficulty.
The all-object late gap is nevertheless substantial. Both chronological
distribution shift and accumulated closed-loop error are possible; frame
stratification cannot separate them. Initial/missing-pose failures remain in
the populations exactly as in native scoring.

The next sampling design should cover later training views and test feedback
stability. It must preserve teacher/student separation: GT-based perturbations
or transport of empirical initialization error are training augmentation only,
never deployment initialization. Existing packed native byte archives do not
cover every later frame, so missing packed members require explicit verified
native-byte fallback or a separate complete cache, not silent early-frame reuse.

Reproducer: `tools/audit_serial_time_coverage.py`. The JSON receipt binds train
initializers, stream registry and both native predictions by SHA256 and states
the physical holdout rule. It performs no model training or GPU work.
