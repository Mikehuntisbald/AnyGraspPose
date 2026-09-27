# V72: matched local-CAD projection training

Candidate changes only the CAD atlas readout relative to the corrected V68r1control: nearest8geometric candidates, local learned choice, and straight-through final-XYZ gradient into the continuous DPT proposal. No new parameters or changed initialization tensors. Source remains V60-extra SHA2560c6f712c7848316259dae3d24554dd4dfac506a20404676e43fb679b94d46d7e. The config persists cad_atlas.local_surface_projection=true; defaultfalse preserves old models.

Reuse V68r1control only after exact replay of its first2updates on8ranks (loss, gradient norm, LR, all metrics, window metadata). Candidate trains200updates from the same source and seeds, with complete-state resume at2. It uses the original predicted DPT prior, not V70/V71oracle XYZ or endpoint interventions. Pose/history remain frozen, DINO feature loss disabled. Geometry/endpoint targets retain their existing real/CAD ownership and valid-key correction.

Physical-holdout paired probes use the same68050000seed sequence:32/64/32frames at0/10/60degrees. Historical baseline/control probes are copied, not recomputed under changed settings. Canonical XYZ, raw-depth error and all adverse slices must be reported. No promotion based solely on flow, oracle, training loss or a single condition. No automatic budget extension or official-test access.

Remote root unified_jepa_20260921/local_projection_joint_v72; immutable runtime /tmp/dexycb_local_projection_joint_v72_r0. Historical control checkpoint stays in surface_joint_v68_r1/control/seed42/last.pt; copied control receipts/logs are not a new control training run.
