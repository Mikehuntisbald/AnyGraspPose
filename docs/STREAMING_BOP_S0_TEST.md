# Frozen residual 1,000: complete causal tracking on s0 test

This experiment evaluates the trained temporal model with persistent source and
context caches. It is separate from the earlier independent-image benchmark.
The checkpoint is unchanged: SHA256
`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`.

The four branches are `lip_temporal`, `lip_no_feature_history`, `fp_tracking`,
and `lip_fp_temporal`. All start with the same earliest legal released PoseCNN
detection, refined twice by frozen FP. Initial observations are encoded for LIP
without changing the common initial output. There is no GT initialization,
redetection after initialization, GT reset, hand annotation input, or future-frame
input. Model/configuration selection is frozen before this run.

Each branch commits its own predictions. LIP+FP commits the LIP proposal and
then applies `model.correct(..., relocalization=False)` to the FP result, keeping
the original source/crop/context history. The no-feature-history branch clears
only source/context caches before each update; its accepted pose, previous pose,
motion features, timestamp and frame counter remain intact. This is a paired
closed-loop history ablation; divergent trajectories are part of its effect.

All intermediate RGB-D frames are processed at the dataset's 30 Hz frame clock.
Only official target frames are exported. The population is 4,916 object streams,
337,862 object-frame updates, 88,014 official target instances in 23,644 images.
Each branch is expected to emit 87,043 predictions. The remaining 971 targets
occur before a legal initialization, including 38 streams without one; they stay
in the scoring denominator. Invalid updates hold the last legal pose and advance
the clock; software exceptions abort. Failures are recorded rather than hidden
by a GT reset.

Official BOP scoring uses the same pinned DexYCB evaluator, VSD/MSSD/MSPD
thresholds and mesh-origin conversion as the independent-image experiment.
Official all-object and grasp-only AR are reported. Temporal inputs mean these
numbers must be identified as a tracking protocol, not placed without qualification
on a single-image leaderboard.

Additional diagnostics use BOP `visib_fract` (<0.5, <0.3, >=0.5, and explicit
empty <0.1), separately for all and grasped targets. Paired 95% bootstrap intervals
resample physical sequences (all eight cameras together), using 1,000 shared
draws with seed 20260913. These are posthoc diagnostics, not the Visibility Aware
paper's hand-projection visibility or ADD AUC. No claim of causal isolation of the
new cross-attention branch is made from this model-versus-FP experiment.

Runtime: `/mnt/why/dexycb_lip/stream_bop_residual_1000_20260913`.
Experiment: `runs/streaming_s0_test/`.

- `protocol.json`: frozen weights, inputs, code hashes and protocol.
- `protocol_tests.log`, `smoke_manifest.json`: actual prelaunch checks.
- `rank*/manifest.json`, `rank*.log`, `rank*/events.jsonl`: real inference progress.
- `status.json`, `supervisor.log`: pipeline status and failures.
- `csv/*.csv`: original-mesh-coordinate submissions after verified merge.
- `<method>/results.json`: official scores when completed.
- `occlusion_ar.json`: visibility metrics and paired intervals when completed.
- `report.md`: final summary only after scoring completes.

The supervisor proceeds from complete inference to official scoring and occlusion
aggregation automatically. The presence of this document does not establish that
the long evaluation has finished; check `status.json` and completion receipts.

## Accelerated scheduling

The execution was repartitioned after a same-input concurrency probe verified 384
exported poses exactly equal to the original run. Four independent processes now
share each H20. The 2,087 verified completed streams (142,955 object frames) are
retained in `retained*/`; original shards and code copies remain available. Only
331 unsealed boundary updates need to be repeated as part of complete stream
replays. The remaining 2,829 streams / 194,907 object frames are assigned to 32
workers with balanced frame loads.

`execution_32.json` records the original protocol hash, assignment hash and new
entrypoint/control hashes. Scheduling changes no pose mathematics, initialization,
history or population. `launch.json` lists both retained completed shards and
live worker folders. The supervisor validates their disjoint union before scoring.
Official scoring uses 16 scene partitions per method; it still concatenates the
official matches and aggregates once. `supervisor_32.log` records the new control
process; `supervisor.log` belongs to the original eight-worker schedule.

## Scoring recovery

All inference completed with the expected population and no recorded invalid
updates. During the 64-process scoring launch, two `xvfb-run -a` allocations
collided at display `:127`; the no-feature-history and FP scoring jobs failed.
The temporal LIP and temporal LIP+FP scores completed successfully and are retained.

The wrapper now starts one private Xvfb per method using `-displayfd`, which
allocates the display atomically, and shares it across independent renderer
contexts. A real probe passed with four distinct displays and 16 simultaneous
GLFW windows. This changes display lifecycle only; official metric code and
renderer remain unchanged. `finish_streaming_bop.py --resume-scoring` verifies
the existing CSV hashes, preserves successful results, archives failed scoring
directories, and reruns only missing scores. Recovery logs are in
`supervisor_recovery.log`; inference is not repeated.
