# V68 matched joint geometry: original versus supervised surface feedback

Training-partition physical holdout; 32 paired observations at0 degrees,64 at10 degrees,32 at60 degrees.
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
| surface_0 | nonheavy | real | 8 | 11.787 | 7.999 |
| surface_0 | nonheavy | proxy | 10 | 13.445 | 11.131 |
| surface_0 | heavy | real | 16 | 11.170 | 10.015 |
| surface_0 | heavy | proxy | 14 | 9.595 | 8.734 |
| baseline_10 | nonheavy | real | 16 | 11.890 | 5.073 |
| baseline_10 | nonheavy | proxy | 16 | 11.665 | 7.593 |
| baseline_10 | heavy | real | 31 | 10.798 | 8.749 |
| baseline_10 | heavy | proxy | 20 | 11.330 | 10.441 |
| control_10 | nonheavy | real | 16 | 10.172 | 5.679 |
| control_10 | nonheavy | proxy | 16 | 11.144 | 9.996 |
| control_10 | heavy | real | 31 | 11.255 | 9.726 |
| control_10 | heavy | proxy | 20 | 11.139 | 9.662 |
| surface_10 | nonheavy | real | 16 | 10.077 | 5.609 |
| surface_10 | nonheavy | proxy | 16 | 10.899 | 9.754 |
| surface_10 | heavy | real | 31 | 11.255 | 9.730 |
| surface_10 | heavy | proxy | 20 | 11.368 | 9.548 |
| baseline_60 | nonheavy | real | 8 | 56.510 | 16.449 |
| baseline_60 | nonheavy | proxy | 10 | 39.461 | 17.186 |
| baseline_60 | heavy | real | 16 | 36.448 | 17.089 |
| baseline_60 | heavy | proxy | 14 | 35.940 | 16.297 |
| control_60 | nonheavy | real | 8 | 55.544 | 13.210 |
| control_60 | nonheavy | proxy | 10 | 38.423 | 18.198 |
| control_60 | heavy | real | 16 | 35.424 | 18.361 |
| control_60 | heavy | proxy | 14 | 37.481 | 18.982 |
| surface_60 | nonheavy | real | 8 | 55.058 | 13.098 |
| surface_60 | nonheavy | proxy | 10 | 37.753 | 17.997 |
| surface_60 | heavy | real | 16 | 34.969 | 18.290 |
| surface_60 | heavy | proxy | 14 | 37.121 | 18.825 |

## Flow endpoint errors in crop pixels

| Variant | Source | Round 0 | Round 1 |
|---|---|---:|---:|
| baseline_0 | observed | 4.710 | 4.465 |
| baseline_0 | real | 4.617 | 4.015 |
| baseline_0 | proxy | 5.179 | 4.263 |
| control_0 | observed | 4.452 | 4.364 |
| control_0 | real | 4.211 | 4.034 |
| control_0 | proxy | 4.903 | 4.278 |
| surface_0 | observed | 4.384 | 4.503 |
| surface_0 | real | 3.917 | 4.275 |
| surface_0 | proxy | 4.647 | 4.426 |
| baseline_10 | observed | 5.868 | 5.573 |
| baseline_10 | real | 6.072 | 5.320 |
| baseline_10 | proxy | 7.871 | 6.465 |
| control_10 | observed | 5.586 | 5.344 |
| control_10 | real | 5.414 | 4.948 |
| control_10 | proxy | 7.220 | 6.096 |
| surface_10 | observed | 5.546 | 5.281 |
| surface_10 | real | 5.575 | 4.888 |
| surface_10 | proxy | 7.162 | 5.781 |
| baseline_60 | observed | 28.437 | 28.069 |
| baseline_60 | real | 24.930 | 24.977 |
| baseline_60 | proxy | 25.432 | 24.545 |
| control_60 | observed | 27.799 | 27.723 |
| control_60 | real | 23.947 | 23.529 |
| control_60 | proxy | 24.399 | 23.720 |
| surface_60 | observed | 28.003 | 26.504 |
| surface_60 | real | 24.963 | 21.786 |
| surface_60 | proxy | 24.530 | 23.747 |

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
| surface_0 | False | observed | 16 | 4.057 |
| surface_0 | False | real | 7 | 3.725 |
| surface_0 | False | proxy | 9 | 4.541 |
| surface_0 | True | observed | 16 | 4.949 |
| surface_0 | True | real | 16 | 4.516 |
| surface_0 | True | proxy | 12 | 4.340 |
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
| surface_10 | False | observed | 31 | 4.878 |
| surface_10 | False | real | 15 | 4.277 |
| surface_10 | False | proxy | 11 | 5.368 |
| surface_10 | True | observed | 30 | 5.696 |
| surface_10 | True | real | 31 | 5.184 |
| surface_10 | True | proxy | 19 | 6.020 |
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
| surface_60 | False | observed | 14 | 25.892 |
| surface_60 | False | real | 6 | 20.397 |
| surface_60 | False | proxy | 7 | 21.229 |
| surface_60 | True | observed | 14 | 27.115 |
| surface_60 | True | real | 16 | 22.306 |
| surface_60 | True | proxy | 10 | 25.509 |

## Paired candidate minus control geometry error

Negative error change is better. Win fraction counts strictly lower error per eligible frame.

| Angle | Heavy | Source | Metric | Frames | Mean change | Win fraction |
|---:|---|---|---|---:|---:|---:|
| 0 | False | real | canonical_xyz_mm | 8 | 0.0010 | 50.0% |
| 0 | False | real | depth_mm | 8 | 0.0474 | 37.5% |
| 0 | False | proxy | canonical_xyz_mm | 10 | -0.2453 | 70.0% |
| 0 | False | proxy | depth_mm | 10 | -0.1934 | 70.0% |
| 0 | True | real | canonical_xyz_mm | 16 | 0.0990 | 37.5% |
| 0 | True | real | depth_mm | 16 | 0.0066 | 50.0% |
| 0 | True | proxy | canonical_xyz_mm | 14 | 0.3434 | 42.9% |
| 0 | True | proxy | depth_mm | 14 | -0.0779 | 57.1% |
| 10 | False | real | canonical_xyz_mm | 16 | -0.0945 | 50.0% |
| 10 | False | real | depth_mm | 16 | -0.0701 | 56.2% |
| 10 | False | proxy | canonical_xyz_mm | 16 | -0.2454 | 50.0% |
| 10 | False | proxy | depth_mm | 16 | -0.2417 | 62.5% |
| 10 | True | real | canonical_xyz_mm | 31 | 0.0001 | 48.4% |
| 10 | True | real | depth_mm | 31 | 0.0045 | 48.4% |
| 10 | True | proxy | canonical_xyz_mm | 20 | 0.2282 | 30.0% |
| 10 | True | proxy | depth_mm | 20 | -0.1140 | 50.0% |
| 60 | False | real | canonical_xyz_mm | 8 | -0.4865 | 75.0% |
| 60 | False | real | depth_mm | 8 | -0.1120 | 62.5% |
| 60 | False | proxy | canonical_xyz_mm | 10 | -0.6697 | 90.0% |
| 60 | False | proxy | depth_mm | 10 | -0.2013 | 50.0% |
| 60 | True | real | canonical_xyz_mm | 16 | -0.4547 | 81.2% |
| 60 | True | real | depth_mm | 16 | -0.0708 | 68.8% |
| 60 | True | proxy | canonical_xyz_mm | 14 | -0.3608 | 50.0% |
| 60 | True | proxy | depth_mm | 14 | -0.1579 | 71.4% |
