# V25: fix detached AMP preview silently freezing shared training parameters

## Verified defect

The CAD RoPE input preparation computes confidence under`torch.no_grad()` using
`core.src_proj` and`visibility`. The main training forward then reuses those
Linear modules inside the SAME enclosing bfloat16 autocast context. On this
PyTorch2.8.0+cu128/H20 environment, the no-grad call caches detached weight casts.
The main forward reuses them, so its shared Linear weight/bias receive NO gradient,
even though their requires_grad flag is true and the optimizer lists them.

This matches the actual V21/V24 checkpoint's absent Adam state for exactly these
four parameters: core.src_proj.weight/bias and visibility.1.weight/bias. Other
unused parameters (history-disabled norms, unselected final DINO block, etc.) are
not classified as this bug. The defect compromises intended shared projection
and visibility learning; its contribution to the entire LIP accuracy gap must be
measured and is not presumed from the gradient finding alone.

## Fix and proof

The detached preview now disables autocast WEIGHT CACHING locally, preserving the
surrounding autocast enable flag and dtype. Confidence remains detached; the
main forward builds gradient-bearing casts normally. No parameter or inference
arithmetic changes at fixed weights.

Actual full-model CUDA comparison with V21 step1000 on an H20 verified:

|Parameter|Old gradient|Fixed gradient norm (probe loss)|
|---|---|---:|
|core.src_proj.weight|None|445.691|
|core.src_proj.bias|None|4.138|
|visibility.1.weight|None|184.446|
|visibility.1.bias|None|13.750|

This connectivity probe's synthetic scalar loss is not a training-quality metric.
Maximum absolute forward difference is EXACTLY0 for patch latent, recovered
features, XYZ, evidence logits and pose. No optimizer updates occurred.

The actual eight-GPU training-objective preflight then verified nonzero gradients
5.928,0.0624,0.01056,0.000744 for those four parameters, as well as pose gradients
through restored XYZ/depth, both feature heads and the shared JEPA patch. CPU
regression tests cover AMP on/off, detached confidence and identical forward
values.11 relevant regression tests passed; no environment packages changed.

## Same-parent performance control now running

Start from the exact V21 step1000 model, optimizer and all-rank RNG. Restore the
LEGACY joint-trained pose head, not V24's frozen shape-conditioned readout. Run
200 updates to1200 with the SAME seed, samples, losses, learning rates, schedule
horizon and feedback-state fix as completed V22 control. Only detached-preview
cache handling changes the training graph. The four previously unused Adam
states begin naturally on their first actual gradient.

After the run: full23,200-frame native validation, fixed40 recovery and controlled
pose probes. Compare against V22 control1200 and V21 parent. Do not adopt on
training loss alone, do not automatically expand this pilot, and do not run an
official test or multiple seeds. Strict1002→1003 full resume is checked first.
The original restoration/pose acceptance goal remains active and unmet.

Runtime `/tmp/dexycb_amp_preview_v25`; output
`/mnt/why/dexycb_lip/unified_jepa_20260921/amp_preview_v25/fixed`.
The controller waited for V24's actual process completion and GPU release.
An initial diagnostic wrapper omitted the serial tracker readout arguments and
failed before training; its logs/source are preserved separately. The corrected
wrapper preserved the serial inputs and passed the full CUDA proof above.

The saved stage-start audit hashes the entire model, Adam state, scheduler and
all-rank RNG against V22 control: all four are identical. After excluding only
the output directory and the new preflight-gradient assertion flag, normalized
configurations are identical as well. This is a paired software-gradient fix,
not a changed learning-rate or loss-weight experiment.
