# JEPA-only fixed32 reconstruction fit

No pose, PnP, flow-readout training or pose evaluation.200 geometry updates;32 fixed training observations/64 hypotheses. Every step checked identical inputs/targets. Strict2→200 resume verified.

|Fixed TRAINING set|CAD identity XYZ real mm|Real sensor depth mm|CAD identity XYZ proxy mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|first_forward|15.766|11.851|14.805|11.919|
|last_forward|6.041|3.780|5.385|3.400|

These are logged pre-update forward metrics (first versus last training step), equal-rank means. They establish fitting ability, not held-out accuracy.

|Independent HEAVY probe|CAD identity XYZ real mm|Real sensor depth mm|CAD identity XYZ proxy mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|0|14.422|12.505|18.336|13.360|
|200|23.678|14.255|27.864|17.944|

Object-coordinate XYZ measures CAD surface identity correspondence. Camera-space recovered geometry must be assessed separately through depth and calibrated camera rays. Real-depth supervision remains original sensor depth; naturally hidden proxy remains GT CAD. No learned decoder was promoted from this memorization diagnostic.
