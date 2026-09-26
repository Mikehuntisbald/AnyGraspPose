# V63 DPT route sensitivity

64 frozen training-partition physical-holdout frames;10/60 degree controlled references. All targets retained.
Gradient shares are input-norm-scaled local sensitivities, not information fractions. XYZ uses the existing straight-through local surrogate.
Roll means a14px horizontal shift of one input lattice; this is a frozen intervention, not a proposed trained model.

| Input / DPT scale | Depth gradient share | XYZ surrogate share | Heavy10 real frames | XYZ output change mm | Depth output change mm |
|---|---:|---:|---:|---:|---:|
| 0 / 64x64 | 52.46% | 36.59% | 15 | 3.039 | 5.431 |
| 1 / 32x32 | 35.99% | 44.84% | 15 | 15.566 | 4.604 |
| 2 / 16x16 | 9.11% | 12.86% | 15 | 4.065 | 1.005 |
| 3 / 8x8 | 2.43% | 5.71% | 15 | 4.799 | 0.469 |

Normal and gradient-capture forward values are bitwise equal to the baseline for all64 frames.
Earlier branches dominate local depth sensitivity. The last branch still affects geometry; the result does not prove it is unused.
