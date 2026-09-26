# V56 serial flow / JEPA prototype

Training-partition physical holdout; 64 paired observations at 10 degrees, 32 at 60 degrees.
GT-perturbed references are a controlled diagnostic, not native rollout or pose accuracy.
Means below are per eligible frame, with original unfiltered real/proxy masks. No confidence filtering.

| Variant | Occlusion | Source | Frames | Canonical XYZ mm | Depth mm |
|---|---|---|---:|---:|---:|
| baseline_v53 | nonheavy | real | 16 | 11.856 | 6.852 |
| baseline_v53 | nonheavy | proxy | 21 | 10.088 | 8.431 |
| baseline_v53 | heavy | real | 32 | 8.857 | 7.383 |
| baseline_v53 | heavy | proxy | 20 | 8.129 | 7.732 |
| initial | nonheavy | real | 16 | 11.850 | 6.854 |
| initial | nonheavy | proxy | 21 | 10.133 | 8.413 |
| initial | heavy | real | 32 | 8.858 | 7.384 |
| initial | heavy | proxy | 20 | 8.132 | 7.727 |
| final_on | nonheavy | real | 16 | 11.823 | 7.153 |
| final_on | nonheavy | proxy | 21 | 9.895 | 9.289 |
| final_on | heavy | real | 32 | 8.695 | 7.617 |
| final_on | heavy | proxy | 20 | 8.846 | 9.052 |
| final_no_feedback | nonheavy | real | 16 | 11.830 | 7.154 |
| final_no_feedback | nonheavy | proxy | 21 | 9.897 | 9.287 |
| final_no_feedback | heavy | real | 32 | 8.700 | 7.616 |
| final_no_feedback | heavy | proxy | 20 | 8.849 | 9.059 |
| final_no_transport | nonheavy | real | 16 | 11.816 | 7.153 |
| final_no_transport | nonheavy | proxy | 21 | 9.911 | 9.307 |
| final_no_transport | heavy | real | 32 | 8.699 | 7.616 |
| final_no_transport | heavy | proxy | 20 | 8.845 | 9.049 |
| large_baseline | nonheavy | real | 8 | 35.559 | 10.522 |
| large_baseline | nonheavy | proxy | 8 | 36.141 | 13.134 |
| large_baseline | heavy | real | 16 | 36.474 | 14.811 |
| large_baseline | heavy | proxy | 10 | 35.165 | 13.809 |
| large_final | nonheavy | real | 8 | 35.189 | 9.686 |
| large_final | nonheavy | proxy | 8 | 35.290 | 13.209 |
| large_final | heavy | real | 16 | 38.143 | 16.756 |
| large_final | heavy | proxy | 10 | 35.376 | 14.673 |

## Flow endpoint errors in crop pixels

| Variant | Source | Round 0 | Round 1 |
|---|---|---:|---:|
| initial | observed | 12.958 | 10.015 |
| initial | real | 26.419 | 23.227 |
| initial | proxy | 23.590 | 20.700 |
| final_on | observed | 6.643 | 5.222 |
| final_on | real | 8.050 | 5.569 |
| final_on | proxy | 8.238 | 5.360 |
| final_no_feedback | observed | 6.643 | 6.691 |
| final_no_feedback | real | 8.050 | 8.030 |
| final_no_feedback | proxy | 8.238 | 8.238 |
| final_no_transport | observed | 6.643 | 5.213 |
| final_no_transport | real | 8.050 | 5.580 |
| final_no_transport | proxy | 8.238 | 5.412 |
| large_final | observed | 26.267 | 25.626 |
| large_final | real | 24.805 | 24.605 |
| large_final | proxy | 21.003 | 20.410 |
