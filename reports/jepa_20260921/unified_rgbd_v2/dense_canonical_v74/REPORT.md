# V74 matched geometry: local CAD fallback versus serial dense canonical readout

Training-partition physical holdout; 32 paired observations at0 degrees,64 at10 degrees,32 at60 degrees.
GT-perturbed references are a controlled diagnostic, not native rollout or pose accuracy.
Means below are per eligible frame, with original unfiltered real/proxy masks. No confidence filtering.

| Variant | Occlusion | Source | Frames | Canonical XYZ mm | Depth mm |
|---|---|---|---:|---:|---:|
| baseline_0 | nonheavy | real | 8 | 11.139 | 5.778 |
| baseline_0 | nonheavy | proxy | 12 | 9.443 | 7.930 |
| baseline_0 | heavy | real | 16 | 8.540 | 7.681 |
| baseline_0 | heavy | proxy | 8 | 11.067 | 9.439 |
| control_0 | nonheavy | real | 8 | 10.368 | 7.263 |
| control_0 | nonheavy | proxy | 12 | 8.079 | 7.664 |
| control_0 | heavy | real | 16 | 8.747 | 8.041 |
| control_0 | heavy | proxy | 8 | 12.425 | 8.702 |
| surface_0 | nonheavy | real | 8 | 0.530 | 7.039 |
| surface_0 | nonheavy | proxy | 12 | 0.577 | 7.710 |
| surface_0 | heavy | real | 16 | 0.600 | 8.024 |
| surface_0 | heavy | proxy | 8 | 1.195 | 8.960 |
| baseline_10 | nonheavy | real | 16 | 10.220 | 9.108 |
| baseline_10 | nonheavy | proxy | 22 | 11.642 | 8.609 |
| baseline_10 | heavy | real | 32 | 9.622 | 8.860 |
| baseline_10 | heavy | proxy | 16 | 11.979 | 8.843 |
| control_10 | nonheavy | real | 16 | 9.343 | 9.842 |
| control_10 | nonheavy | proxy | 22 | 9.879 | 8.192 |
| control_10 | heavy | real | 32 | 10.007 | 8.633 |
| control_10 | heavy | proxy | 16 | 12.152 | 7.968 |
| surface_10 | nonheavy | real | 16 | 8.402 | 9.563 |
| surface_10 | nonheavy | proxy | 22 | 7.539 | 8.130 |
| surface_10 | heavy | real | 32 | 9.412 | 8.611 |
| surface_10 | heavy | proxy | 16 | 9.061 | 7.981 |
| baseline_60 | nonheavy | real | 8 | 40.805 | 19.189 |
| baseline_60 | nonheavy | proxy | 12 | 41.334 | 23.133 |
| baseline_60 | heavy | real | 15 | 34.245 | 15.759 |
| baseline_60 | heavy | proxy | 8 | 36.912 | 18.596 |
| control_60 | nonheavy | real | 8 | 44.234 | 16.458 |
| control_60 | nonheavy | proxy | 12 | 39.158 | 22.402 |
| control_60 | heavy | real | 15 | 34.258 | 16.057 |
| control_60 | heavy | proxy | 8 | 38.838 | 15.064 |
| surface_60 | nonheavy | real | 8 | 65.369 | 16.190 |
| surface_60 | nonheavy | proxy | 12 | 47.997 | 22.775 |
| surface_60 | heavy | real | 15 | 39.791 | 16.180 |
| surface_60 | heavy | proxy | 8 | 39.157 | 14.492 |

## Flow endpoint errors in crop pixels

| Variant | Source | Round 0 | Round 1 |
|---|---|---:|---:|
| baseline_0 | observed | 3.727 | 3.769 |
| baseline_0 | real | 3.932 | 3.487 |
| baseline_0 | proxy | 4.791 | 3.967 |
| control_0 | observed | 3.772 | 3.733 |
| control_0 | real | 3.744 | 3.448 |
| control_0 | proxy | 3.818 | 3.436 |
| surface_0 | observed | 3.869 | 3.729 |
| surface_0 | real | 3.833 | 3.366 |
| surface_0 | proxy | 3.977 | 3.466 |
| baseline_10 | observed | 5.029 | 4.834 |
| baseline_10 | real | 5.147 | 4.348 |
| baseline_10 | proxy | 6.883 | 6.346 |
| control_10 | observed | 4.817 | 4.584 |
| control_10 | real | 4.786 | 4.387 |
| control_10 | proxy | 6.381 | 5.890 |
| surface_10 | observed | 4.818 | 4.648 |
| surface_10 | real | 4.792 | 4.400 |
| surface_10 | proxy | 6.549 | 5.938 |
| baseline_60 | observed | 27.672 | 27.187 |
| baseline_60 | real | 28.070 | 26.853 |
| baseline_60 | proxy | 25.694 | 24.691 |
| control_60 | observed | 28.013 | 27.495 |
| control_60 | real | 28.255 | 27.703 |
| control_60 | proxy | 25.308 | 24.911 |
| surface_60 | observed | 27.904 | 28.507 |
| surface_60 | real | 28.179 | 28.587 |
| surface_60 | proxy | 24.628 | 25.774 |

## Flow by requested occlusion condition

| Variant | Heavy | Source | Frames | Round 1 EPE px |
|---|---|---|---:|---:|
| baseline_0 | False | observed | 16 | 3.362 |
| baseline_0 | False | real | 5 | 2.886 |
| baseline_0 | False | proxy | 11 | 3.628 |
| baseline_0 | True | observed | 14 | 4.234 |
| baseline_0 | True | real | 16 | 3.675 |
| baseline_0 | True | proxy | 6 | 4.590 |
| control_0 | False | observed | 16 | 3.431 |
| control_0 | False | real | 5 | 3.153 |
| control_0 | False | proxy | 11 | 2.864 |
| control_0 | True | observed | 14 | 4.077 |
| control_0 | True | real | 16 | 3.540 |
| control_0 | True | proxy | 6 | 4.484 |
| surface_0 | False | observed | 16 | 3.472 |
| surface_0 | False | real | 5 | 3.150 |
| surface_0 | False | proxy | 11 | 2.915 |
| surface_0 | True | observed | 14 | 4.022 |
| surface_0 | True | real | 16 | 3.434 |
| surface_0 | True | proxy | 6 | 4.478 |
| baseline_10 | False | observed | 29 | 4.965 |
| baseline_10 | False | real | 14 | 3.795 |
| baseline_10 | False | proxy | 17 | 5.981 |
| baseline_10 | True | observed | 29 | 4.703 |
| baseline_10 | True | real | 31 | 4.598 |
| baseline_10 | True | proxy | 11 | 6.909 |
| control_10 | False | observed | 29 | 4.740 |
| control_10 | False | real | 14 | 3.598 |
| control_10 | False | proxy | 17 | 5.691 |
| control_10 | True | observed | 29 | 4.429 |
| control_10 | True | real | 31 | 4.744 |
| control_10 | True | proxy | 11 | 6.196 |
| surface_10 | False | observed | 29 | 4.749 |
| surface_10 | False | real | 14 | 3.852 |
| surface_10 | False | proxy | 17 | 5.828 |
| surface_10 | True | observed | 29 | 4.548 |
| surface_10 | True | real | 31 | 4.647 |
| surface_10 | True | proxy | 11 | 6.108 |
| baseline_60 | False | observed | 16 | 25.144 |
| baseline_60 | False | real | 7 | 33.422 |
| baseline_60 | False | proxy | 8 | 26.531 |
| baseline_60 | True | observed | 14 | 29.521 |
| baseline_60 | True | real | 15 | 23.788 |
| baseline_60 | True | proxy | 4 | 21.012 |
| control_60 | False | observed | 16 | 25.283 |
| control_60 | False | real | 7 | 36.063 |
| control_60 | False | proxy | 8 | 27.181 |
| control_60 | True | observed | 14 | 30.024 |
| control_60 | True | real | 15 | 23.802 |
| control_60 | True | proxy | 4 | 20.371 |
| surface_60 | False | observed | 16 | 26.739 |
| surface_60 | False | real | 7 | 37.188 |
| surface_60 | False | proxy | 8 | 28.247 |
| surface_60 | True | observed | 14 | 30.527 |
| surface_60 | True | real | 15 | 24.573 |
| surface_60 | True | proxy | 4 | 20.827 |

## Paired candidate minus control geometry error

Negative error change is better. Win fraction counts strictly lower error per eligible frame.

| Angle | Heavy | Source | Metric | Frames | Mean change | Win fraction |
|---:|---|---|---|---:|---:|---:|
| 0 | False | real | canonical_xyz_mm | 8 | -9.8386 | 100.0% |
| 0 | False | real | depth_mm | 8 | -0.2235 | 75.0% |
| 0 | False | proxy | canonical_xyz_mm | 12 | -7.5023 | 100.0% |
| 0 | False | proxy | depth_mm | 12 | 0.0467 | 41.7% |
| 0 | True | real | canonical_xyz_mm | 16 | -8.1468 | 100.0% |
| 0 | True | real | depth_mm | 16 | -0.0174 | 43.8% |
| 0 | True | proxy | canonical_xyz_mm | 8 | -11.2306 | 100.0% |
| 0 | True | proxy | depth_mm | 8 | 0.2586 | 37.5% |
| 10 | False | real | canonical_xyz_mm | 16 | -0.9412 | 56.2% |
| 10 | False | real | depth_mm | 16 | -0.2788 | 50.0% |
| 10 | False | proxy | canonical_xyz_mm | 22 | -2.3406 | 54.5% |
| 10 | False | proxy | depth_mm | 22 | -0.0614 | 59.1% |
| 10 | True | real | canonical_xyz_mm | 32 | -0.5951 | 59.4% |
| 10 | True | real | depth_mm | 32 | -0.0220 | 65.6% |
| 10 | True | proxy | canonical_xyz_mm | 16 | -3.0913 | 62.5% |
| 10 | True | proxy | depth_mm | 16 | 0.0131 | 50.0% |
| 60 | False | real | canonical_xyz_mm | 8 | 21.1348 | 12.5% |
| 60 | False | real | depth_mm | 8 | -0.2683 | 75.0% |
| 60 | False | proxy | canonical_xyz_mm | 12 | 8.8389 | 33.3% |
| 60 | False | proxy | depth_mm | 12 | 0.3735 | 33.3% |
| 60 | True | real | canonical_xyz_mm | 15 | 5.5335 | 26.7% |
| 60 | True | real | depth_mm | 15 | 0.1236 | 46.7% |
| 60 | True | proxy | canonical_xyz_mm | 8 | 0.3184 | 50.0% |
| 60 | True | proxy | depth_mm | 8 | -0.5716 | 75.0% |
