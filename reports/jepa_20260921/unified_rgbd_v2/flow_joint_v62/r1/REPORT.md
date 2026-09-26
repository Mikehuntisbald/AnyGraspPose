# V62 matched joint geometry: write gain1 versus100

Training-partition physical holdout; 32 paired observations at0 degrees,64 at10 degrees,32 at60 degrees.
GT-perturbed references are a controlled diagnostic, not native rollout or pose accuracy.
Means below are per eligible frame, with original unfiltered real/proxy masks. No confidence filtering.

| Variant | Occlusion | Source | Frames | Canonical XYZ mm | Depth mm |
|---|---|---|---:|---:|---:|
| baseline_0 | nonheavy | real | 8 | 8.699 | 7.029 |
| baseline_0 | nonheavy | proxy | 11 | 8.987 | 9.059 |
| baseline_0 | heavy | real | 16 | 8.838 | 12.127 |
| baseline_0 | heavy | proxy | 11 | 9.390 | 7.134 |
| control_0 | nonheavy | real | 8 | 8.642 | 7.293 |
| control_0 | nonheavy | proxy | 11 | 8.798 | 9.913 |
| control_0 | heavy | real | 16 | 9.157 | 12.796 |
| control_0 | heavy | proxy | 11 | 8.958 | 8.115 |
| strong_0 | nonheavy | real | 8 | 8.622 | 7.213 |
| strong_0 | nonheavy | proxy | 11 | 8.824 | 10.199 |
| strong_0 | heavy | real | 16 | 9.212 | 12.738 |
| strong_0 | heavy | proxy | 11 | 8.945 | 8.192 |
| baseline_10 | nonheavy | real | 16 | 8.385 | 6.529 |
| baseline_10 | nonheavy | proxy | 21 | 9.809 | 9.926 |
| baseline_10 | heavy | real | 32 | 10.029 | 10.837 |
| baseline_10 | heavy | proxy | 23 | 10.191 | 8.344 |
| control_10 | nonheavy | real | 16 | 8.780 | 8.250 |
| control_10 | nonheavy | proxy | 21 | 10.557 | 9.168 |
| control_10 | heavy | real | 32 | 10.063 | 11.262 |
| control_10 | heavy | proxy | 23 | 10.121 | 8.899 |
| strong_10 | nonheavy | real | 16 | 8.688 | 8.137 |
| strong_10 | nonheavy | proxy | 21 | 10.551 | 9.239 |
| strong_10 | heavy | real | 32 | 10.092 | 11.250 |
| strong_10 | heavy | proxy | 23 | 10.140 | 9.041 |
| baseline_60 | nonheavy | real | 8 | 41.523 | 10.483 |
| baseline_60 | nonheavy | proxy | 11 | 28.789 | 16.874 |
| baseline_60 | heavy | real | 16 | 35.829 | 19.623 |
| baseline_60 | heavy | proxy | 10 | 42.505 | 22.031 |
| control_60 | nonheavy | real | 8 | 41.575 | 14.131 |
| control_60 | nonheavy | proxy | 11 | 27.916 | 18.697 |
| control_60 | heavy | real | 16 | 37.410 | 20.973 |
| control_60 | heavy | proxy | 10 | 43.273 | 22.072 |
| strong_60 | nonheavy | real | 8 | 41.456 | 14.645 |
| strong_60 | nonheavy | proxy | 11 | 27.929 | 18.727 |
| strong_60 | heavy | real | 16 | 37.431 | 20.939 |
| strong_60 | heavy | proxy | 10 | 43.359 | 22.130 |

## Flow endpoint errors in crop pixels

| Variant | Source | Round 0 | Round 1 |
|---|---|---:|---:|
| baseline_0 | observed | 4.298 | 3.931 |
| baseline_0 | real | 4.732 | 3.870 |
| baseline_0 | proxy | 5.201 | 4.086 |
| control_0 | observed | 4.365 | 4.068 |
| control_0 | real | 4.013 | 3.629 |
| control_0 | proxy | 4.503 | 3.691 |
| strong_0 | observed | 4.331 | 4.138 |
| strong_0 | real | 4.152 | 3.609 |
| strong_0 | proxy | 4.735 | 3.816 |
| baseline_10 | observed | 5.148 | 4.825 |
| baseline_10 | real | 6.345 | 5.260 |
| baseline_10 | proxy | 6.066 | 5.082 |
| control_10 | observed | 5.060 | 4.756 |
| control_10 | real | 5.569 | 5.015 |
| control_10 | proxy | 5.493 | 4.807 |
| strong_10 | observed | 5.007 | 4.806 |
| strong_10 | real | 5.539 | 4.975 |
| strong_10 | proxy | 5.564 | 4.836 |
| baseline_60 | observed | 25.784 | 25.539 |
| baseline_60 | real | 24.735 | 24.831 |
| baseline_60 | proxy | 24.058 | 23.779 |
| control_60 | observed | 24.655 | 24.649 |
| control_60 | real | 24.436 | 24.541 |
| control_60 | proxy | 23.877 | 22.424 |
| strong_60 | observed | 24.717 | 24.801 |
| strong_60 | real | 24.450 | 24.444 |
| strong_60 | proxy | 23.200 | 22.568 |
