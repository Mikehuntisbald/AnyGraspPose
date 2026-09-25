# V19 joint JEPA and pose continuation

Source: step35400, SHA256 `d0846bf32dfe43919e93b2172ae5c0e017afea77fd4b014a1fab1075d837e3fc`. Target45400, exactly10000 additional updates, seed42,8H20,effective batch32. History remains disabled.

Unfreeze the full pose readout: query, object attention/norm, geometry relation readout and pose head. Keep the trained encoder/JEPA/DPT/CAD modules trainable. No architecture change, parallel branch or weight reset. Original pose loss (normalized translation SmoothL1 +0.5 rotation angle + normalized transformed-point error) has weight1; every previous recovery term remains unchanged. Pose loss passes through the same recovered geometry and JEPA patch.

Peaks: encoder3e-7, predictor3e-6, new modules3e-5, pose readout3e-6. Existing modules start at inherited terminal0.1 peak, then200-step linear rewarm and cosine decay over10000 steps. Old Adam moments/steps inherited by parameter name; newly unfrozen readout states empty. All model/EMA tensors, RNG and sampler inherited.

Retain the immutable causal crop reference to preserve batched training throughput and input distribution. Student predictions do not drive training crops; native validation uses the student feedback loop. This distinction is explicit and pose efficacy requires native validation, not training loss.

Validation: eight-rank startup/resume audit after3 updates; source35400 native val; native val and fixed40 recovery at36400,40400,45400. Native val320 streams/23200 frames, history off, same initializers. Preserve full checkpoints every50 steps and milestones. No official test, multi-seed or default-model promotion.

Execution: isolated local-disk source under `/tmp/dexycb_joint_pose_v19_local`; durable artifacts under `/mnt/why/dexycb_lip/unified_jepa_20260921/joint_pose_v19`. Source archive and hash manifest pinned before training. No edits to prior experiments.

Verified launch: joint H20 preflight passed; pose-only gradients reached patch/XYZ/depth (norms0.02156/1.16e-6/2.77e-6 on one train frame; connectivity evidence, not efficacy). Full40-frame joint backward finite; 485 existing Adam tensor states inherited,22 readout tensors newly trainable; peak10.68GB. Eight-rank35403 audit passed with pose and JEPA weights changed, history unchanged, EMA continuous.

Native packaging initially missed the inference entrypoint/runtime. Smoke then identified the static CAD models_info allowance and scalar diagnostic outputs; fixed without changing student/source code or training checkpoint provenance. Exact metadata-only allowance tested; inference and scoring smoke passed on2 sequences/6 frames, zero rejected updates, zero GT pose/mask reads in inference. Smoke is not a benchmark score. Full native source35400 then training to45400 resumed automatically.
