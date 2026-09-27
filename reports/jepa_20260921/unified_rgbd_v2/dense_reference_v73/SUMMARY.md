# V73: useful dense CAD geometry is not preserved by current reconstruction

Frozen V72-fast source;96physical-holdout training-partition frames,32each at0/10/60degree initial rotation error. All variants use identical target pixels and reference coverage. Missing reference pixels keep the network fallback. Zero-error references are controlled diagnostics, not a deployed GT input.

| Heavy region | Network XYZ mm | Dense estimated reference | Patch sample | Patch mean |
|---|---:|---:|---:|---:|
| 0deg real-hidden |9.860|0.000|11.549|10.709|
| 0deg CAD proxy |8.781|0.000|11.075|9.487|
| 10deg real-hidden |14.156|10.784|19.545|14.913|
| 10deg CAD proxy |10.843|6.854|13.073|11.747|
| 60deg real-hidden |35.861|38.298|38.364|38.097|
| 60deg CAD proxy |49.780|48.406|49.819|48.269|

At0degrees, dense input canonical geometry is correct while the network adds roughly9–10mm error. Explicit patch-coordinate representations also lose detail. Patch sample includes the model's patch-validity rule; patch mean uses exact pixel validity. These are coarse-reference baselines, not evidence that the learned256D latent is literally a mean coordinate or a proof of one sole bottleneck.

Dense reference is useful at small initialization errors but not uniformly at60degrees. It must be aligned and allowed to fall back. Depth must remain separate: even0degree CAD geometry has12.488mm discrepancy against real-hidden raw sensor depth, versus11.755mm for the network; proxy rendered depth is0mm at0degrees. Do not relabel real depth as CAD depth or claim copying rendered depth solves reconstruction.

V74implements a JEPA-conditioned dense backward correspondence/readout at both reconstruction stages. It preserves full-resolution CAD XYZ and independent depth/validity, keeps full-CAD fallback where lookup is unsupported, and receives independently projected dense correspondence/support targets. A bounded paired200update experiment is running; no default promotion or achieved-accuracy claim.
