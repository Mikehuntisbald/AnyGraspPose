# V57 frozen flow / depth / CAD geometry audit

V56 weights frozen. 64 controlled10-degree frames and32 controlled60-degree frames.
Training-partition physical holdout; no native/pose evaluation. PnP/3D fitting are diagnostics only.
Means are per eligible frame on original target masks; unavailable candidate pixels retain the original prediction.
Oracle branches use labels and must never be interpreted as deployed accuracy.

| Rotation | Occlusion | Method | Source | Frames | XYZ mm | Depth mm | Candidate coverage |
|---:|---|---|---|---:|---:|---:|---:|
| 10 | nonheavy | baseline | real | 16 | 11.823 | 7.153 | n/a |
| 10 | nonheavy | baseline | proxy | 21 | 9.895 | 9.289 | n/a |
| 10 | heavy | baseline | real | 32 | 8.695 | 7.617 | n/a |
| 10 | heavy | baseline | proxy | 20 | 8.846 | 9.052 | n/a |
| 10 | nonheavy | triangle | real | 16 | 10.843 | 7.153 | 0.517 |
| 10 | nonheavy | triangle | proxy | 21 | 9.920 | 9.289 | 0.403 |
| 10 | heavy | triangle | real | 32 | 8.459 | 7.617 | 0.563 |
| 10 | heavy | triangle | proxy | 20 | 8.949 | 9.052 | 0.408 |
| 10 | nonheavy | rigid | real | 16 | 8.445 | 51.848 | 0.927 |
| 10 | nonheavy | rigid | proxy | 21 | 8.277 | 56.451 | 0.918 |
| 10 | heavy | rigid | real | 32 | 7.952 | 29.849 | 0.877 |
| 10 | heavy | rigid | proxy | 20 | 8.226 | 32.624 | 0.816 |
| 10 | nonheavy | metric_recovered | real | 16 | 8.609 | 7.464 | 0.748 |
| 10 | nonheavy | metric_recovered | proxy | 21 | 7.497 | 8.338 | 0.902 |
| 10 | heavy | metric_recovered | real | 32 | 6.547 | 8.626 | 0.860 |
| 10 | heavy | metric_recovered | proxy | 20 | 6.616 | 7.426 | 0.803 |
| 10 | nonheavy | metric_measured | real | 16 | 8.605 | 7.551 | 0.809 |
| 10 | nonheavy | metric_measured | proxy | 21 | 7.473 | 9.413 | 0.901 |
| 10 | heavy | metric_measured | real | 32 | 6.807 | 8.413 | 0.795 |
| 10 | heavy | metric_measured | proxy | 20 | 7.179 | 9.268 | 0.730 |
| 10 | nonheavy | oracle_triangle | real | 16 | 7.645 | 7.153 | 0.433 |
| 10 | nonheavy | oracle_triangle | proxy | 21 | 7.718 | 9.289 | 0.272 |
| 10 | heavy | oracle_triangle | real | 32 | 5.549 | 7.617 | 0.442 |
| 10 | heavy | oracle_triangle | proxy | 20 | 7.670 | 9.052 | 0.200 |
| 10 | nonheavy | oracle_rigid | real | 16 | 1.768 | 10.437 | 0.812 |
| 10 | nonheavy | oracle_rigid | proxy | 21 | 0.560 | 0.164 | 0.952 |
| 10 | heavy | oracle_rigid | real | 32 | 0.840 | 10.070 | 0.875 |
| 10 | heavy | oracle_rigid | proxy | 20 | 0.729 | 0.577 | 0.900 |
| 10 | nonheavy | oracle_cad | real | 16 | 0.000 | 11.600 | 1.000 |
| 10 | nonheavy | oracle_cad | proxy | 21 | 0.000 | 0.000 | 1.000 |
| 10 | heavy | oracle_cad | real | 32 | 0.000 | 10.234 | 1.000 |
| 10 | heavy | oracle_cad | proxy | 20 | 0.000 | 0.000 | 1.000 |
| 60 | nonheavy | baseline | real | 8 | 35.189 | 9.686 | n/a |
| 60 | nonheavy | baseline | proxy | 8 | 35.290 | 13.209 | n/a |
| 60 | heavy | baseline | real | 16 | 38.143 | 16.756 | n/a |
| 60 | heavy | baseline | proxy | 10 | 35.376 | 14.673 | n/a |
| 60 | nonheavy | triangle | real | 8 | 36.356 | 9.686 | 0.599 |
| 60 | nonheavy | triangle | proxy | 8 | 32.986 | 13.209 | 0.490 |
| 60 | heavy | triangle | real | 16 | 39.918 | 16.756 | 0.577 |
| 60 | heavy | triangle | proxy | 10 | 35.630 | 14.673 | 0.127 |
| 60 | nonheavy | rigid | real | 8 | 36.296 | 57.509 | 0.730 |
| 60 | nonheavy | rigid | proxy | 8 | 33.436 | 54.618 | 0.833 |
| 60 | heavy | rigid | real | 16 | 40.211 | 35.751 | 0.741 |
| 60 | heavy | rigid | proxy | 10 | 35.944 | 25.636 | 0.517 |
| 60 | nonheavy | metric_recovered | real | 8 | 35.226 | 10.080 | 0.375 |
| 60 | nonheavy | metric_recovered | proxy | 8 | 34.509 | 13.975 | 0.293 |
| 60 | heavy | metric_recovered | real | 16 | 39.193 | 18.168 | 0.650 |
| 60 | heavy | metric_recovered | proxy | 10 | 36.039 | 16.348 | 0.319 |
| 60 | nonheavy | metric_measured | real | 8 | 36.752 | 13.067 | 0.375 |
| 60 | nonheavy | metric_measured | proxy | 8 | 34.226 | 12.488 | 0.178 |
| 60 | heavy | metric_measured | real | 16 | 39.466 | 18.163 | 0.651 |
| 60 | heavy | metric_measured | proxy | 10 | 36.343 | 17.002 | 0.314 |
| 60 | nonheavy | oracle_triangle | real | 8 | 21.161 | 9.686 | 0.410 |
| 60 | nonheavy | oracle_triangle | proxy | 8 | 27.134 | 13.209 | 0.180 |
| 60 | heavy | oracle_triangle | real | 16 | 27.605 | 16.756 | 0.326 |
| 60 | heavy | oracle_triangle | proxy | 10 | 34.426 | 14.673 | 0.054 |
| 60 | nonheavy | oracle_rigid | real | 8 | 2.823 | 12.749 | 0.875 |
| 60 | nonheavy | oracle_rigid | proxy | 8 | 4.470 | 3.398 | 0.875 |
| 60 | heavy | oracle_rigid | real | 16 | 2.086 | 10.288 | 0.875 |
| 60 | heavy | oracle_rigid | proxy | 10 | 2.876 | 2.668 | 0.800 |
| 60 | nonheavy | oracle_cad | real | 8 | 0.000 | 13.432 | 1.000 |
| 60 | nonheavy | oracle_cad | proxy | 8 | 0.000 | 0.000 | 1.000 |
| 60 | heavy | oracle_cad | real | 16 | 0.000 | 9.875 | 1.000 |
| 60 | heavy | oracle_cad | proxy | 10 | 0.000 | 0.000 | 1.000 |
