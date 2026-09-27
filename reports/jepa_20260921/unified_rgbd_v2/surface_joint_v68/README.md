# V68 interrupted: masked-target correction required

This run is terminal/interrupted, not an active training job. Control completed200updates; candidate completed97 before deliberate controller termination. All artifacts are preserved remotely under unified_jepa_20260921/surface_joint_v68. This local folder contains compact receipts/logs only; full checkpoints and per-step rank logs remain remote.

New recovery correspondence CE reached574.33 for an observed region at step14: Gaussian target tails lay on invalid crop keys whose logits are fixed to-10000. That contribution is unlearnable and invalidates interpreting the total loss. Existing appearance CE used the same unconditioned target form. This is an implementation defect in the objective, not established as the root cause of all earlier geometry failures.

V68r1 conditions soft correspondence targets on valid keys and renormalizes for BOTH appearance and recovery CE. Endpoint and geometry masks/labels remain unchanged. An explicit single-valid-key test requires zero matching CE and zero score gradient;11focused tests passed. Both arms restart from the original identical V60-extra tensors under the corrected objective,200updates each. Do not compare an interrupted candidate checkpoint against the completed control as the claimed matched result.
