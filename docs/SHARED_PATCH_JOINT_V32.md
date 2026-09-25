# V32: joint pose training on the unified JEPA representation

Status: implemented; paired experiment launched. No accuracy claim or model
promotion until native and restoration results are collected.

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
