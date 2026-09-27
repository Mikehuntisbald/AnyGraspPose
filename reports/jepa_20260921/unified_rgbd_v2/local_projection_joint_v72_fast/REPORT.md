# V72 matched geometry: original versus local CAD projection

Training-partition physical holdout; 32 paired observations at0 degrees,64 at10 degrees,32 at60 degrees.
Historical V68r1 control200 is reused after exact first2-update replay on8ranks. Candidate trained200updates.
GT-perturbed references are a controlled diagnostic, not native rollout or pose accuracy.
Means below are per eligible frame, with original unfiltered real/proxy masks. No confidence filtering.

| Variant | Occlusion | Source | Frames | Canonical XYZ mm | Depth mm |
|---|---|---|---:|---:|---:|
| baseline_0 | nonheavy | real | 8 | 12.676 | 7.487 |
| baseline_0 | nonheavy | proxy | 10 | 13.309 | 8.240 |
| baseline_0 | heavy | real | 16 | 10.663 | 10.108 |
| baseline_0 | heavy | proxy | 14 | 9.848 | 8.897 |
| control_0 | nonheavy | real | 8 | 11.786 | 7.952 |
| control_0 | nonheavy | proxy | 10 | 13.690 | 11.324 |
| control_0 | heavy | real | 16 | 11.071 | 10.008 |
| control_0 | heavy | proxy | 14 | 9.252 | 8.812 |
| surface_0 | nonheavy | real | 8 | 12.874 | 7.999 |
| surface_0 | nonheavy | proxy | 10 | 12.726 | 11.161 |
| surface_0 | heavy | real | 16 | 10.267 | 9.901 |
| surface_0 | heavy | proxy | 14 | 8.559 | 8.393 |
| baseline_10 | nonheavy | real | 16 | 11.890 | 5.073 |
| baseline_10 | nonheavy | proxy | 16 | 11.665 | 7.593 |
| baseline_10 | heavy | real | 31 | 10.798 | 8.749 |
| baseline_10 | heavy | proxy | 20 | 11.330 | 10.441 |
| control_10 | nonheavy | real | 16 | 10.172 | 5.679 |
| control_10 | nonheavy | proxy | 16 | 11.144 | 9.996 |
| control_10 | heavy | real | 31 | 11.255 | 9.726 |
| control_10 | heavy | proxy | 20 | 11.139 | 9.662 |
| surface_10 | nonheavy | real | 16 | 9.732 | 5.656 |
| surface_10 | nonheavy | proxy | 16 | 10.330 | 9.908 |
| surface_10 | heavy | real | 31 | 10.770 | 9.691 |
| surface_10 | heavy | proxy | 20 | 11.370 | 9.344 |
| baseline_60 | nonheavy | real | 8 | 56.510 | 16.449 |
| baseline_60 | nonheavy | proxy | 10 | 39.461 | 17.186 |
| baseline_60 | heavy | real | 16 | 36.448 | 17.089 |
| baseline_60 | heavy | proxy | 14 | 35.940 | 16.297 |
| control_60 | nonheavy | real | 8 | 55.544 | 13.210 |
| control_60 | nonheavy | proxy | 10 | 38.423 | 18.198 |
| control_60 | heavy | real | 16 | 35.424 | 18.361 |
| control_60 | heavy | proxy | 14 | 37.481 | 18.982 |
| surface_60 | nonheavy | real | 8 | 54.041 | 13.525 |
| surface_60 | nonheavy | proxy | 10 | 37.042 | 18.214 |
| surface_60 | heavy | real | 16 | 35.278 | 18.160 |
| surface_60 | heavy | proxy | 14 | 36.822 | 18.731 |

## Flow endpoint errors in crop pixels

| Variant | Source | Round 0 | Round 1 |
|---|---|---:|---:|
| baseline_0 | observed | 4.710 | 4.465 |
| baseline_0 | real | 4.617 | 4.015 |
| baseline_0 | proxy | 5.179 | 4.263 |
| control_0 | observed | 4.452 | 4.364 |
| control_0 | real | 4.211 | 4.034 |
| control_0 | proxy | 4.903 | 4.278 |
| surface_0 | observed | 4.467 | 4.352 |
| surface_0 | real | 4.153 | 3.987 |
| surface_0 | proxy | 4.872 | 4.258 |
| baseline_10 | observed | 5.868 | 5.573 |
| baseline_10 | real | 6.072 | 5.320 |
| baseline_10 | proxy | 7.871 | 6.465 |
| control_10 | observed | 5.586 | 5.344 |
| control_10 | real | 5.414 | 4.948 |
| control_10 | proxy | 7.220 | 6.096 |
| surface_10 | observed | 5.561 | 5.333 |
| surface_10 | real | 5.448 | 4.985 |
| surface_10 | proxy | 7.312 | 6.073 |
| baseline_60 | observed | 28.437 | 28.069 |
| baseline_60 | real | 24.930 | 24.977 |
| baseline_60 | proxy | 25.432 | 24.545 |
| control_60 | observed | 27.799 | 27.723 |
| control_60 | real | 23.947 | 23.529 |
| control_60 | proxy | 24.399 | 23.720 |
| surface_60 | observed | 27.719 | 27.732 |
| surface_60 | real | 24.236 | 24.048 |
| surface_60 | proxy | 24.424 | 23.849 |

## Flow by requested occlusion condition

| Variant | Heavy | Source | Frames | Round 1 EPE px |
|---|---|---|---:|---:|
| baseline_0 | False | observed | 16 | 4.067 |
| baseline_0 | False | real | 7 | 3.776 |
| baseline_0 | False | proxy | 9 | 5.309 |
| baseline_0 | True | observed | 16 | 4.864 |
| baseline_0 | True | real | 16 | 4.120 |
| baseline_0 | True | proxy | 12 | 3.479 |
| control_0 | False | observed | 16 | 3.836 |
| control_0 | False | real | 7 | 3.709 |
| control_0 | False | proxy | 9 | 4.554 |
| control_0 | True | observed | 16 | 4.891 |
| control_0 | True | real | 16 | 4.176 |
| control_0 | True | proxy | 12 | 4.071 |
| surface_0 | False | observed | 16 | 3.860 |
| surface_0 | False | real | 7 | 3.647 |
| surface_0 | False | proxy | 9 | 4.543 |
| surface_0 | True | observed | 16 | 4.843 |
| surface_0 | True | real | 16 | 4.136 |
| surface_0 | True | proxy | 12 | 4.044 |
| baseline_10 | False | observed | 31 | 5.646 |
| baseline_10 | False | real | 15 | 4.992 |
| baseline_10 | False | proxy | 11 | 6.459 |
| baseline_10 | True | observed | 30 | 5.497 |
| baseline_10 | True | real | 31 | 5.478 |
| baseline_10 | True | proxy | 19 | 6.468 |
| control_10 | False | observed | 31 | 4.930 |
| control_10 | False | real | 15 | 4.496 |
| control_10 | False | proxy | 11 | 5.572 |
| control_10 | True | observed | 30 | 5.773 |
| control_10 | True | real | 31 | 5.167 |
| control_10 | True | proxy | 19 | 6.400 |
| surface_10 | False | observed | 31 | 4.973 |
| surface_10 | False | real | 15 | 4.544 |
| surface_10 | False | proxy | 11 | 5.570 |
| surface_10 | True | observed | 30 | 5.705 |
| surface_10 | True | real | 31 | 5.199 |
| surface_10 | True | proxy | 19 | 6.364 |
| baseline_60 | False | observed | 14 | 29.043 |
| baseline_60 | False | real | 6 | 29.516 |
| baseline_60 | False | proxy | 7 | 22.427 |
| baseline_60 | True | observed | 14 | 27.095 |
| baseline_60 | True | real | 16 | 23.274 |
| baseline_60 | True | proxy | 10 | 26.027 |
| control_60 | False | observed | 14 | 28.262 |
| control_60 | False | real | 6 | 26.647 |
| control_60 | False | proxy | 7 | 21.126 |
| control_60 | True | observed | 14 | 27.183 |
| control_60 | True | real | 16 | 22.360 |
| control_60 | True | proxy | 10 | 25.537 |
| surface_60 | False | observed | 14 | 28.263 |
| surface_60 | False | real | 6 | 28.323 |
| surface_60 | False | proxy | 7 | 21.160 |
| surface_60 | True | observed | 14 | 27.201 |
| surface_60 | True | real | 16 | 22.445 |
| surface_60 | True | proxy | 10 | 25.732 |

## Paired candidate minus control geometry error

Negative error change is better. Win fraction counts strictly lower error per eligible frame.

| Angle | Heavy | Source | Metric | Frames | Mean change | Win fraction |
|---:|---|---|---|---:|---:|---:|
| 0 | False | real | canonical_xyz_mm | 8 | 1.0880 | 50.0% |
| 0 | False | real | depth_mm | 8 | 0.0471 | 25.0% |
| 0 | False | proxy | canonical_xyz_mm | 10 | -0.9645 | 70.0% |
| 0 | False | proxy | depth_mm | 10 | -0.1627 | 70.0% |
| 0 | True | real | canonical_xyz_mm | 16 | -0.8048 | 87.5% |
| 0 | True | real | depth_mm | 16 | -0.1067 | 50.0% |
| 0 | True | proxy | canonical_xyz_mm | 14 | -0.6931 | 78.6% |
| 0 | True | proxy | depth_mm | 14 | -0.4185 | 78.6% |
| 10 | False | real | canonical_xyz_mm | 16 | -0.4392 | 62.5% |
| 10 | False | real | depth_mm | 16 | -0.0231 | 50.0% |
| 10 | False | proxy | canonical_xyz_mm | 16 | -0.8141 | 81.2% |
| 10 | False | proxy | depth_mm | 16 | -0.0878 | 62.5% |
| 10 | True | real | canonical_xyz_mm | 31 | -0.4856 | 77.4% |
| 10 | True | real | depth_mm | 31 | -0.0342 | 45.2% |
| 10 | True | proxy | canonical_xyz_mm | 20 | 0.2309 | 55.0% |
| 10 | True | proxy | depth_mm | 20 | -0.3177 | 65.0% |
| 60 | False | real | canonical_xyz_mm | 8 | -1.5036 | 75.0% |
| 60 | False | real | depth_mm | 8 | 0.3153 | 50.0% |
| 60 | False | proxy | canonical_xyz_mm | 10 | -1.3814 | 60.0% |
| 60 | False | proxy | depth_mm | 10 | 0.0164 | 40.0% |
| 60 | True | real | canonical_xyz_mm | 16 | -0.1461 | 50.0% |
| 60 | True | real | depth_mm | 16 | -0.2005 | 68.8% |
| 60 | True | proxy | canonical_xyz_mm | 14 | -0.6591 | 71.4% |
| 60 | True | proxy | depth_mm | 14 | -0.2513 | 71.4% |
