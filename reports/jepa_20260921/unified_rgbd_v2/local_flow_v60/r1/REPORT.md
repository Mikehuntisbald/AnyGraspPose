# V60 local flow matched readout: full forward results

Training-partition physical holdout; 64 paired observations at 10 degrees, 32 at 60 degrees.
GT-perturbed references are a controlled diagnostic, not native rollout or pose accuracy.
Means below are per eligible frame, with original unfiltered real/proxy masks. No confidence filtering.

| Variant | Occlusion | Source | Frames | Canonical XYZ mm | Depth mm |
|---|---|---|---:|---:|---:|
| baseline_10 | nonheavy | real | 16 | 9.119 | 6.541 |
| baseline_10 | nonheavy | proxy | 20 | 9.419 | 6.655 |
| baseline_10 | heavy | real | 30 | 11.284 | 8.850 |
| baseline_10 | heavy | proxy | 20 | 13.660 | 9.613 |
| control_10 | nonheavy | real | 16 | 9.123 | 6.541 |
| control_10 | nonheavy | proxy | 20 | 9.425 | 6.652 |
| control_10 | heavy | real | 30 | 11.277 | 8.846 |
| control_10 | heavy | proxy | 20 | 13.661 | 9.606 |
| extra_10 | nonheavy | real | 16 | 9.121 | 6.544 |
| extra_10 | nonheavy | proxy | 20 | 9.430 | 6.660 |
| extra_10 | heavy | real | 30 | 11.285 | 8.847 |
| extra_10 | heavy | proxy | 20 | 13.672 | 9.612 |
| baseline_60 | nonheavy | real | 8 | 35.715 | 23.948 |
| baseline_60 | nonheavy | proxy | 12 | 21.912 | 15.646 |
| baseline_60 | heavy | real | 15 | 42.778 | 20.933 |
| baseline_60 | heavy | proxy | 11 | 48.379 | 20.490 |
| control_60 | nonheavy | real | 8 | 35.711 | 23.952 |
| control_60 | nonheavy | proxy | 12 | 21.909 | 15.650 |
| control_60 | heavy | real | 15 | 42.775 | 20.929 |
| control_60 | heavy | proxy | 11 | 48.389 | 20.479 |
| extra_60 | nonheavy | real | 8 | 35.725 | 23.949 |
| extra_60 | nonheavy | proxy | 12 | 21.921 | 15.650 |
| extra_60 | heavy | real | 15 | 42.775 | 20.930 |
| extra_60 | heavy | proxy | 11 | 48.390 | 20.490 |

## Flow endpoint errors in crop pixels

| Variant | Source | Round 0 | Round 1 |
|---|---|---:|---:|
| baseline_10 | observed | 7.426 | 5.862 |
| baseline_10 | real | 8.860 | 6.742 |
| baseline_10 | proxy | 9.164 | 5.655 |
| control_10 | observed | 6.222 | 5.536 |
| control_10 | real | 7.395 | 6.046 |
| control_10 | proxy | 7.961 | 6.043 |
| extra_10 | observed | 5.419 | 4.994 |
| extra_10 | real | 6.100 | 5.224 |
| extra_10 | proxy | 6.266 | 5.235 |
| baseline_60 | observed | 28.502 | 27.672 |
| baseline_60 | real | 31.852 | 30.045 |
| baseline_60 | proxy | 26.513 | 25.864 |
| control_60 | observed | 27.792 | 27.326 |
| control_60 | real | 31.893 | 30.606 |
| control_60 | proxy | 26.911 | 26.462 |
| extra_60 | observed | 28.268 | 27.975 |
| extra_60 | real | 31.942 | 30.952 |
| extra_60 | proxy | 27.409 | 26.980 |
