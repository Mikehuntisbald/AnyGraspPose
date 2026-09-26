# V32: joint pose training on the unified JEPA representation

Status: paired500-update training and all validations completed. The candidate
does not beat the control on heavy pose or improve geometric restoration;
neither is promoted to the default model. The original LIP target is unmet.

## Scope

The requested route is observation/CAD appearance and geometry → JEPA patch
latent → object query → pose, with restoration useful to pose. The later
decoded-only restriction was narrower than that original route. V27 tested
fresh heads on frozen outputs; its failure does not test whether joint pose
training can shape the actual shared representation.

V32 keeps the DINO4/11 decoded features and DPT canonical XYZ/depth/validity
readout. It adds a zero-initialized projection of the SAME final JEPA patch to
the128-channel appearance embedding, before combining it with the128-channel
geometric relation and reading a single object query. This is not an FP or
encoder bypass, a second predictor, or a parallel pose estimate. The auxiliary
projection is `LayerNorm256 → Linear256→128`, with zero Linear weight/bias.
Both pose routes originate in the same JEPA patch. All existing feature and
geometry recovery objectives remain in place.

Decoded geometry still supplies measured/predicted point ownership, local
canonical-to-camera residuals, and global relation moments. Actual measured
depth is retained as before. V29's dense selector and V31's aggregate mass cap
remain disabled: neither passed the frozen-pose gate. No GT content enters the
student or readout. History remains disabled under the existing permission.

## Paired trial

- Parent: V26 step1200, SHA256
  `159c50048f08ffaca4111080ccb83a80cfb4967c9227bbbf39741b52223fc977`.
- Both arms contain identical new projection tensors and start with identical
  existing model/EMA, named Adam state, scheduler, and all eight rank RNGs.
  New adapter Adam states begin empty. The existing pose head is not reset.
- Control multiplies the new projection by0; candidate by1. At initialization
  both are exactly the old readout. The only paired config differences are
  output directory and this fixed scalar.
- Both jointly train on actual predicted representations. Oracle-geometry
  readout initialization and rehearsal are disabled. This removal is shared by
  both arms; compare candidate against the contemporaneous control, not solely
  against the older oracle-rehearsal parent.
- Seed42, eight H20, effective batch32,500 additional updates per arm,
 1200→1700. Parent boundary LR factor0.9522542486 is preserved; warm to peak
  over50 updates, then cosine to0.5 at1700. Peak rates remain new1e-4,
  predictor1e-5, pose1e-4, encoder3e-7.
- Complete checkpoints every50 updates and explicit1202→1203 process restart.
  Native23,200-frame validation at1450/1700; stop an arm at1450 if all-frame
  ADD-S@0.05d regresses by more than2 percentage points from parent50.8249%.
  Fixed40 restoration and controlled pose at the terminal1700 checkpoint.
  Candidate1700 also receives same-weight native patch-off evaluation.
- Training and validation use GPUs sequentially. No additional seed, official
  test, package change, or automatic budget extension.

Fourteen targeted tests pass, including zero-adapter exact parity, gradients
through patch AND decoded features/XYZ/depth, named Adam migration with changed
parameter ordering, legacy serial behavior and the AMP preview repair. Eight-GPU
preflight additionally checks the new projection gradient is zero in control
and nonzero in the candidate; existing completion paths must remain active.
Strict resume verifies complete state restoration before advancing.

Activation of a gradient is not evidence of useful restoration. Final acceptance
still requires actual pose/restoration gains and the original pure-LIP
comparison; a successful patch-only shortcut inside JEPA would not establish
that completed geometry helps. Controlled completion-off/appearance-off and
same-weight patch-off results will be reported separately.

Runtime: `/tmp/dexycb_shared_patch_v32`; artifact root:
`/mnt/why/dexycb_lip/unified_jepa_20260921/shared_patch_joint_v32`.
Remote runtime is a pinned source copy, not a Git checkout.

## Verified launch snapshot (not terminal results)

Both eight-GPU preflights pass. The new projection's actual-loss gradient norm
is0 in control and0.80736 in candidate, while decoded feature/XYZ/depth and
shared-patch gradients remain nonzero. Migration verifies695 existing model
tensors and455 existing named Adam states. The1202→1203 process restart passed
strict restoration checks. `paired_start_identity.json` confirms both arms'
model, optimizer, scheduler, RNG, step and sampler position are identical.

Control1450 full native validation: all52.2201%, visibility<50%22.1558%,
visibility<30%7.6848%. Parent1200 was50.8249%/21.9471%/7.7194%. This is the
control without oracle rehearsal, NOT a gain attributable to direct patch
reading. The candidate is training; final conclusions remain pending.

An independent CPU audit found that this inherited serial curriculum reaches
only18.53% of frames in eligible training streams, all no later than absolute
frame28. See `SERIAL_TIME_COVERAGE_AUDIT.md`. V32 leaves sampling unchanged;
coverage and closed-loop stability must be repaired/tested independently.

## Completed results

|1700-step model|All ADD-S@0.05d|Visibility<50%|Visibility<30%|
|---|---:|---:|---:|
|Control, actual-prediction training|56.908%|31.699%|10.892%|
|Shared patch + decoded content|57.202%|30.747%|8.154%|
|Same shared weights, direct patch disabled|57.074%|27.334%|8.213%|

The candidate gains0.293pp overall but loses0.952pp heavy and2.737pp extreme
versus the equal-budget control. Patch-on versus same-weight patch-off gains
3.413pp heavy: the model uses the path, but that sensitivity does not establish
better restoration or superiority to a model trained without that path.

Fixed40 heavy restoration (control → candidate): real XYZ39.402→39.509mm,
depth17.304→17.395mm; CAD-proxy XYZ38.058→38.075mm,
depth23.596→23.728mm. No geometric restoration improvement is demonstrated.
Native initializer/frame populations and recovery crops/teacher/donors/masks
were checked equal. Same-weight patch-off retains identical checkpoint and
config hashes, with a separately recorded inference intervention and zero GT
pose/mask/reset reads.

The first patch-off attempt correctly failed the checkpoint/config identity
gate because an alternate config had been supplied. Training and all earlier
evaluations had already completed. The follow-up uses the original config and
an explicit `--disable-shared-patch` switch, preserving that gate. Original
failure logs and the corrected finish receipt are retained; no training was
restarted and no model weights were edited for the intervention.

Terminal checkpoint SHA256:

- Control: `d8bce096bd96a19dd25656268a2444257ff5df842255f3bbb1207998bcd721ba`.
- Candidate: `3d6c6f8fe19cfbf50aa7b3d55c208baa9242145fdeb0fe7eab5b70ad72228ee1`.

The next sampler comparison uses the control parent because it has better
heavy/extreme pose and no worse reconstruction. This is experimental lineage,
not default-model promotion. `outcome.json` binds the completed comparisons.
