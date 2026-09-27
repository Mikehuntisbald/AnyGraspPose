# V66 geometry-to-flow feedback diagnostic

Frozen V60-extra checkpoint; 64 training-partition physical-holdout frames.
GT substitutions are diagnostics only, not deployable predictions. Original 14px pooling and feedback strength retained.
Heavy means requested augmentation. Frame means; unchanged target masks, no confidence filtering.

| Angle | Heavy | Region | Variant | Frames | Round 1 EPE px | Round 2 EPE px |
|---:|---|---|---|---:|---:|---:|
| 10 | False | observed | normal | 16 | 5.4001 | 5.0424 |
| 10 | False | observed | feedback_off | 16 | 5.4001 | 5.4185 |
| 10 | False | observed | oracle_xyz | 16 | 5.4001 | 4.7763 |
| 10 | False | observed | oracle_xyz_validity | 16 | 5.4001 | 4.7030 |
| 10 | False | real | normal | 8 | 6.9397 | 6.2685 |
| 10 | False | real | feedback_off | 8 | 6.9397 | 6.8523 |
| 10 | False | real | oracle_xyz | 8 | 6.9397 | 5.3462 |
| 10 | False | real | oracle_xyz_validity | 8 | 6.9397 | 5.2295 |
| 10 | False | proxy | normal | 7 | 5.2907 | 4.2166 |
| 10 | False | proxy | feedback_off | 7 | 5.2907 | 5.2864 |
| 10 | False | proxy | oracle_xyz | 7 | 5.2907 | 4.1646 |
| 10 | False | proxy | oracle_xyz_validity | 7 | 5.2907 | 3.9246 |
| 10 | True | observed | normal | 13 | 5.3404 | 5.1764 |
| 10 | True | observed | feedback_off | 13 | 5.3404 | 5.3364 |
| 10 | True | observed | oracle_xyz | 13 | 5.3404 | 4.8008 |
| 10 | True | observed | oracle_xyz_validity | 13 | 5.3404 | 5.0499 |
| 10 | True | real | normal | 16 | 6.7052 | 5.8033 |
| 10 | True | real | feedback_off | 16 | 6.7052 | 6.6862 |
| 10 | True | real | oracle_xyz | 16 | 6.7052 | 5.4324 |
| 10 | True | real | oracle_xyz_validity | 16 | 6.7052 | 5.2285 |
| 10 | True | proxy | normal | 2 | 1.7066 | 2.6687 |
| 10 | True | proxy | feedback_off | 2 | 1.7066 | 1.6940 |
| 10 | True | proxy | oracle_xyz | 2 | 1.7066 | 2.5155 |
| 10 | True | proxy | oracle_xyz_validity | 2 | 1.7066 | 2.5071 |
| 60 | False | observed | normal | 15 | 29.4004 | 29.0987 |
| 60 | False | observed | feedback_off | 15 | 29.4004 | 29.4019 |
| 60 | False | observed | oracle_xyz | 15 | 29.4004 | 26.5278 |
| 60 | False | observed | oracle_xyz_validity | 15 | 29.4004 | 25.4185 |
| 60 | False | real | normal | 7 | 21.1489 | 20.6338 |
| 60 | False | real | feedback_off | 7 | 21.1489 | 21.1473 |
| 60 | False | real | oracle_xyz | 7 | 21.1489 | 19.1552 |
| 60 | False | real | oracle_xyz_validity | 7 | 21.1489 | 16.7905 |
| 60 | False | proxy | normal | 5 | 31.3365 | 32.1371 |
| 60 | False | proxy | feedback_off | 5 | 31.3365 | 31.3336 |
| 60 | False | proxy | oracle_xyz | 5 | 31.3365 | 28.8118 |
| 60 | False | proxy | oracle_xyz_validity | 5 | 31.3365 | 27.8198 |
| 60 | True | observed | normal | 13 | 19.5636 | 19.4722 |
| 60 | True | observed | feedback_off | 13 | 19.5636 | 19.5699 |
| 60 | True | observed | oracle_xyz | 13 | 19.5636 | 18.6714 |
| 60 | True | observed | oracle_xyz_validity | 13 | 19.5636 | 18.9000 |
| 60 | True | real | normal | 14 | 27.5862 | 27.7115 |
| 60 | True | real | feedback_off | 14 | 27.5862 | 27.4487 |
| 60 | True | real | oracle_xyz | 14 | 27.5862 | 27.1641 |
| 60 | True | real | oracle_xyz_validity | 14 | 27.5862 | 27.4665 |
| 60 | True | proxy | normal | 5 | 13.2305 | 12.3995 |
| 60 | True | proxy | feedback_off | 5 | 13.2305 | 13.2311 |
| 60 | True | proxy | oracle_xyz | 5 | 13.2305 | 10.6859 |
| 60 | True | proxy | oracle_xyz_validity | 5 | 13.2305 | 11.9452 |
