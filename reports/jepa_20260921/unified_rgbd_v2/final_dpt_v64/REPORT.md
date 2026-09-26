# V64 final-latent-only DPT routing

200 candidate updates from the same V60 source as V62 strong200. Same data, gain100, objective and schedule.
Historical control replay reproduced first2 updates exactly across8 ranks; full200-step control is reused, not rerun.
Physical training holdout;32/64/32 controlled observations at0/10/60 degrees; no confidence filtering of target regions.

| Angle | Model | Source | Heavy frames | Canonical XYZ mm | Depth mm | Flow EPE px (all eligible frames) |
|---:|---|---|---:|---:|---:|---:|
| 0 | baseline | real | 16 | 8.838 | 12.127 | 3.870 |
| 0 | baseline | proxy | 11 | 9.390 | 7.134 | 4.086 |
| 0 | matched_control | real | 16 | 9.212 | 12.738 | 3.609 |
| 0 | matched_control | proxy | 11 | 8.945 | 8.192 | 3.816 |
| 0 | final_only | real | 16 | 9.122 | 12.658 | 3.531 |
| 0 | final_only | proxy | 11 | 8.880 | 7.916 | 3.790 |
| 10 | baseline | real | 32 | 10.029 | 10.837 | 5.260 |
| 10 | baseline | proxy | 23 | 10.191 | 8.344 | 5.082 |
| 10 | matched_control | real | 32 | 10.092 | 11.250 | 4.975 |
| 10 | matched_control | proxy | 23 | 10.140 | 9.041 | 4.836 |
| 10 | final_only | real | 32 | 9.986 | 11.061 | 4.922 |
| 10 | final_only | proxy | 23 | 10.138 | 9.195 | 4.966 |
| 60 | baseline | real | 16 | 35.829 | 19.623 | 24.831 |
| 60 | baseline | proxy | 10 | 42.505 | 22.031 | 23.779 |
| 60 | matched_control | real | 16 | 37.431 | 20.939 | 24.444 |
| 60 | matched_control | proxy | 10 | 43.359 | 22.130 | 22.568 |
| 60 | final_only | real | 16 | 37.105 | 20.927 | 24.572 |
| 60 | final_only | proxy | 10 | 43.560 | 21.659 | 22.558 |
