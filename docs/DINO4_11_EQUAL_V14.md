# DINO4/11 equal-weight JEPA-only trial

User explicitly requested1000 further JEPA-only optimizer updates and student input4/11 as well as teacher supervision4/11. Continue step10000→11000, seed42, eight H20, effective batch32, four episodes perGPU and eight independent frames per forward batch. No history reads, memory writes or replay loss. No pose loss or pose/query/readout training; no default-model promotion or official test.

Exact source checkpoint SHA256: `6af7a1a258f529800a8751036dce84d04652e19eeb4703e84d10a4f8110599ff`. All source tensors are retained, including both recovery readouts, online DINO and EMA teacher; EMA starts at counter250 and momentum0.996 is retained. This is not a new random initialization. The input/target change uses fresh AdamW and a25-step warmup over the new1000-update horizon. Encoder LR1e-6, predictor1e-5, localCAD/new modules1e-4. RNG and sampler320000 are preserved. Complete checkpoints every50 updates, strict process-restart startup gate10002→10003.

## Layer contract

`dino_layers.student: [4,11]` and `teacher: [4,11]` are1-based. Official calls use `n=[3,10], norm=True`; CLS/register tokens excluded. `mid` compatibility fields now mean block4; `last` means block11. Both observed and estimated-pose CAD appearance inputs use the student4/11 encoder. Teacher real/CAD feature targets use the separate EMA4/11 encoder. The immutable crop-reference model retains original frozen6/12 features and its own causal history.

Feature reconstruction, spatial centering and soft feature correspondence all use `0.625 L4 + 0.625 L11`, retaining the previous coefficient sum1.25. Real/CAD source weighting and XYZ/depth/validity/physical-CAD correspondence coefficients remain unchanged. Unmasked visible pixels still receive no completion targets. The feature-error estimator continues to estimate the deeper head's error, now block11; it is a detached error-estimation target, not an extra block11 feature reconstruction term.

## Validation and interpretation

39 CPU checks passed, including1-based layer extraction, equal reconstruction-head gradients, equal spatial-loss gradients and legacy defaults. H20 B4/T40 preflight verifies exact official layer selection on both student and teacher, original6/12 reference, both restoration-head gradients, online encoder gradients, no pose/history gradients, no unused block12 gradient, stop-gradient teacher, exact EMA formula and complete tensor restore. Its one disposable in-memory optimizer update is never persisted and does not advance the1000-update budget.

Fixed40 sequence evaluation runs on the switched-layer initial state, step10500 and11000, with separate block4/block11 spatial diagnostics and teacher-object-mean baselines. Geometry and localCAD correspondence retain fixed physical targets. Native EMA feature targets evolve during training: lower feature loss alone is not a pose or spatial-recovery guarantee. This trial changes both input and target layers, so its outcome cannot isolate their individual effects.
