# V68: supervised surface feedback, paired 200-update training

V67 has a large oracle-versus-prediction gap. The original global flow CE supervises appearance scores only. Local endpoint gradients reach recovered XYZ, but do not directly teach the global geometry score to select the correct image patch.

The candidate uses V67 surface feedback at strength8 and adds global geometry-match CE from independently projected correspondence targets. Targets/masks remain the audited observed, artificially hidden real and CAD-proxy split; proxy has factor0.5. Geometry CE weight0.05 sits inside the existing flow objective (outer weight0.2). Geometry validity used in the score is detached and remains independently supervised. There are no teacher inputs and no predicted-confidence evaluation exclusions.

Both control and candidate copy every tensor from V60-extra SHA256 0c6f712c7848316259dae3d24554dd4dfac506a20404676e43fb679b94d46d7e. Neither introduces new parameters. Both reset optimizer and train200 updates from identical seeds/data/LR, with pose/history frozen and DINO feature loss disabled. The source step counter is200, while the exported flow head previously received1000 updates; these are different histories.

The controller runs the required tests and baseline probes, then serially trains control and surface arms. Each runs2 steps, saves complete state, resumes to200, and evaluates paired0/10/60-degree physical-holdout cases inside the training partition. The sampler seed is68000000 and probe seed68050000. Both training and validation own eight H20 GPUs exclusively. No official test, additional seeds, extended budget, or default promotion occurs.

Require original supervision audit, nonzero gradients in both serial dependencies, complete-state resume, and frame-paired geometry/flow metrics before conclusions. Runtime source is immutable under /tmp/dexycb_surface_joint_v68_r0; source/config archives and training provenance identify the exact run. Remote project-root copies are not Git alignment.

## Corrected run V68r1

Original V68 stopped after control200/candidate97 because Gaussian soft-target tails contributed fixed masked-logit penalties (recovery CE574.33 at candidate step14). Retain the original artifacts as interrupted. The corrected objective conditions the target distribution on available keys for both appearance and geometry CE; endpoint/geometry supervision and evaluation regions do not change. A one-valid-key regression test requires zero correspondence CE/gradient. Restart both arms from the original source; the corrected runtime is /tmp/dexycb_surface_joint_v68_r1_r0 and artifacts are unified_jepa_20260921/surface_joint_v68_r1. This is a bug-fix rerun, not an extension of the interrupted checkpoints or evidence of improved recovery.
