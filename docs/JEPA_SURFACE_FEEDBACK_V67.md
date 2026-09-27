# Surface-preserving reconstruction-to-flow feedback

V66 established that recovered geometry can help matching, but even oracle XYZ through the existing feedback leaves large rotation/heavy-occlusion errors. V67 tests a concrete candidate rather than claiming this is a proven pooling root cause.

The existing feedback averages XYZ within every 14x14 patch, then penalizes distance to the resulting point. A patch containing multiple surfaces can therefore represent an unreal surface. The candidate samples 16 actual predicted pixels at offsets {1,5,8,12} in each axis, compares each canonical CAD point to those samples, and uses the maximum validity-weighted Gaussian similarity as nonnegative matching evidence. Canonical coordinates remain diameter-normalized; sigma is 0.1 diameter. Predicted validity is detached. Nonfinite candidates supply no evidence. Raw RGB-D, learned appearance scores, estimated-position prior and the local endpoint head are retained.

This changes both the surface aggregation and the evidence form; it is not an isolated attribution to pooling alone. Strengths 2 and 8 are fixed diagnostic arms. Default `surface_feedback_strength=None` retains the old computation and adds no checkpoint parameters. No default model is promoted. The selector is currently a runtime experiment option, not a persisted training configuration.

Frozen V60-extra, exact V66 sample seeds and masks: 64 physical-holdout frames within the training partition. Compare normal, predicted geometry strengths 2/8, oracle XYZ strengths 2/8, and oracle XYZ+validity strengths 2/8. Oracle substitutions are diagnostic only and never a deployed metric. All first-round outputs must be identical; the normal forward must match baseline bitwise. Report all slices, including regressions and sparse proxy coverage.

Tests cover phantom mean surfaces, nonfinite/invalid geometry, geometry gradients with detached validity, chunk equivalence, existing transport gradients and missing-reference behavior. No pose solver, pose training, official test or additional seed is introduced.
