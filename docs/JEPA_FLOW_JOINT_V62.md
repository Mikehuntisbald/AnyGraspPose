# V62: matched joint geometry training with stronger flow writes

V60 improved local correspondences without a stable geometry gain. V61 frozen
interventions found normal flow writes only0.0677%/0.0687% of patch norms in the
two rounds. Addition occurs inFP32, so BF16 addition truncation is not the cause.
Replacing known supported endpoints by GT locations barely alters reconstruction;
100x write gain changes reconstructed coordinates by roughly1mm but does not make
geometry accurate without training. Oracle interventions are diagnostics only.

Two V62 arms start from the same V60 extra export, SHA256
`0c6f712c7848316259dae3d24554dd4dfac506a20404676e43fb679b94d46d7e`.
All original tensors, including the trained local flow head, load exactly. Both
arms use fresh AdamW state and200 bounded updates. The only between-arm model
setting is flow transport gain: control1, strong100. Gain is configuration,
not a secretly edited checkpoint tensor. Neither arm replaces the default.

Both arms use identical seeds/data/losses, observation batch32 on8H20s with mirrored
estimated-pose pairs. Every10th update uses correct references, explicitly
supervising near-zero correspondence changes. Every fourth update uses transported
empirical training initializer errors even in paired mode, addressing the prior
exclusion of large empirical errors. Ground truth creates perturbations and labels
only; it never enters the student tensors as corrected flow or completed geometry.

Train existing encoder/JEPA, flow/local-refinement/writer, DPT and atlas jointly;
pose modules and history remain frozen/off. Existing geometry ownership is
unchanged. Flow objective weight0.2, flow LR3e-4, atlas LR1e-4; other module rates
follow the source geometry configuration. The original and strong arms share all
these settings,20-step warmup, clip1 and the same budget. Stop after2 and strictly
resume to200 with complete optimizer/RNG state. Save every50 steps.

Frozen paired evaluations use32 correct-reference,64 ten-degree and32 sixty-degree
cases for the baseline and each arm. Report original real/proxy geometry targets
and flow together, including regressions. These are controlled physical-holdout
training-partition probes, not native val or pose accuracy. No official test,
additional seed, automatic budget extension or default model promotion.

Outcome: both200-update arms completed without a stable geometry improvement.
See [paired results and limitations](../reports/jepa_20260921/unified_rgbd_v2/flow_joint_v62/SUMMARY.md).
