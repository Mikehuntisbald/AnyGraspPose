# V33: full training-sequence coverage, unchanged JEPA model and losses

Status: both500-update arms and all evaluations completed. No default model changed.

The inherited serial pose loop consumed only relative frames0/1/8/9 after an
early native initializer. In its actual training partition this reached at most
18.53% of frames, none after absolute frame28. V26 native accuracy deteriorated
from75.15% in frames0–28 to34.62% later. This motivates a controlled data repair,
but does not distinguish coverage failure from accumulated tracking drift.

## Sampling and label boundaries

The candidate uniformly selects a legal10-frame window anywhere in the chosen
training stream. Supervised offsets0/1/8/9 can now reach every frame, including
the final frame. The control retains native-initializer-prefix sampling.
Both decode10 frames rather than12; the two removed frames were unused by all
losses and feedback. Stream selection retains the original initializer-option
population and probabilities. Start sampling uses an independent seeded RNG,
so stream, donor and perturbation RNG draws are not shifted by the new draw.

The native initial pose was measured at a different frame. It is not silently
treated as a valid later-frame pose. For TRAINING augmentation only, transport
its rotation error and object-center translation error to the selected window:

`E_R = R_est_source R_gt_source^T`, `e_t = t_est_source - t_gt_source`;
`R_est_window = E_R R_gt_window`, `t_est_window = t_gt_window + e_t`.

Existing perturbations are included in that error. GT transport is confined to
training pose augmentation, analogous to the existing GT-noisy/zero examples;
it is never called by native inference and is not described as a new native
initializer. GT labels remain separate from observed RGB-D and the main model.
Window timestamps use actual absolute frame numbers, so feedback time remains
correct. Sampler provenance and supervised absolute frame IDs are logged.

Packed native bytes are used when all requested members exist. Missing later
members fall back explicitly to original native JPEG/PNG/NPZ files, with missing
files raising an error. There is no early-frame substitution, generated depth,
precomputed pose-conditioned feature or changed CAD cache.

## Paired training

- Parent: V32 control1700, SHA256
  `d8bce096bd96a19dd25656268a2444257ff5df842255f3bbb1207998bcd721ba`.
- Same model/EMA, Adam, RNG, losses and peak learning rates; no new model
  parameters. Direct shared-patch contribution remains disabled in both arms.
- Seed42, eight H20, effective batch32,500 updates per arm,1700→2200.
  Parent boundary LR factor0.5 is retained; warm to peak over50 steps, cosine
  to0.5 by2200. No oracle-geometry rehearsal or history.
- Explicit1702→1703 strict resume, checkpoint every50, native23,200-frame
  validation at1950/2200. Stop an arm at1950 for >2pp overall regression from
  parent56.9085%. Fixed40 recovery and controlled pose at2200.
- Compare full windows against the same-budget prefix control. Do not attribute
  gains from extra updates alone to the sampler. No automatic budget extension,
  additional seed, official test, package change or default-model replacement.

Seventeen targeted tests pass, including complete reachable-frame coverage,
error transport, packed/raw equality, existing readout gradients and AMP repair.
Real-data preflight compares8 actual later windows with their original native
bytes and prefix error distributions;34 of80 frames require raw fallback.
RGB, depth and visible masks match exactly; error transport preserves rotation
and centered translation within2e-6. All8 windows are after absolute frame28.
The formal preflight uses the actual microbatch4 and checks completion/pose
gradient paths. Resume and paired starting-state receipts will be collected.

Runtime `/tmp/dexycb_full_window_v33`; artifacts
`/mnt/why/dexycb_lip/unified_jepa_20260921/full_window_v33`.
The remote runtime is a pinned source copy, not a Git checkout.

## Verified startup and midpoint

Both8-GPU preflights and1702→1703 strict restoration pass. The paired-start
receipt verifies identical model, optimizer, scheduler, RNG, step and sampler
position; only output directory and window mode differ between configs.
Observed rank0 sampled frames extend to73 in the full-window arm, with60.75%
after28 in the captured launch snapshot, versus0% in the prefix arm. This is
finite observed coverage, separate from the support of the sampling scheme.

At1950 (+250 updates), full-window all/heavy/extreme ADD-S@0.05d is
54.933%/22.577%/8.225%, versus prefix55.567%/29.515%/10.129%.
The sampler repair has NOT yet improved pose. Overall regression from the
parent is1.975pp, below the declared >2pp stop threshold; both arms continue
only within the fixed remaining250-update budget. Final conclusions are pending.

## Terminal2200 results

|Sampling|All ADD-S@0.05d|Visibility<50%|Visibility<30%|
|---|---:|---:|---:|
|Prefix|59.0074%|27.0179%|10.1081%|
|Full window|59.0093%|27.5730%|13.3011%|

The final full-window arm is effectively tied overall,+0.555pp heavy,+3.193pp
extreme. It does not establish a geometry-restoration improvement. Fixed40
heavy real XYZ/depth is39.545/16.064mm prefix versus39.883/15.979mm full;
proxy is39.313/23.124mm versus39.573/23.539mm. The user's accuracy goal is unmet.
The next short geometry-only experiment retains full temporal training support
but uses independent current-frame initialization, and compares model changes
on a training-partition physical holdout before further native validation.
