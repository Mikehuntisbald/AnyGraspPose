# V69 frozen flow-aligned canonical prior

64 physical-holdout training-partition frames, frozen V60-extra. Same seeds as V66/V67.
Oracle endpoints and oracle-supported anchor selection are diagnostic only. No training or pose solver.
Flow/depth are bitwise unchanged. All original target pixels remain in evaluation; no coverage filtering.

| Angle | Heavy | Region | Variant | Frames | Canonical XYZ mm | Depth mm |
|---:|---|---|---|---:|---:|---:|
| 10 | False | real | normal | 8 | 15.3640 | 8.5988 |
| 10 | False | real | pred14 | 8 | 15.4907 | 8.5988 |
| 10 | False | real | oracle14 | 8 | 13.3520 | 8.5988 |
| 10 | False | real | oracle_supported14 | 8 | 13.2335 | 8.5988 |
| 10 | False | proxy | normal | 12 | 11.3600 | 8.5379 |
| 10 | False | proxy | pred14 | 12 | 10.6363 | 8.5379 |
| 10 | False | proxy | oracle14 | 12 | 10.4111 | 8.5379 |
| 10 | False | proxy | oracle_supported14 | 12 | 11.0681 | 8.5379 |
| 10 | True | real | normal | 16 | 13.4582 | 12.1016 |
| 10 | True | real | pred14 | 16 | 12.8905 | 12.1016 |
| 10 | True | real | oracle14 | 16 | 11.4316 | 12.1016 |
| 10 | True | real | oracle_supported14 | 16 | 11.3523 | 12.1016 |
| 10 | True | proxy | normal | 6 | 15.1400 | 15.4902 |
| 10 | True | proxy | pred14 | 6 | 15.7839 | 15.4902 |
| 10 | True | proxy | oracle14 | 6 | 14.1653 | 15.4902 |
| 10 | True | proxy | oracle_supported14 | 6 | 20.6638 | 15.4902 |
| 60 | False | real | normal | 7 | 27.0436 | 15.8586 |
| 60 | False | real | pred14 | 7 | 27.9312 | 15.8586 |
| 60 | False | real | oracle14 | 7 | 19.1885 | 15.8586 |
| 60 | False | real | oracle_supported14 | 7 | 15.2838 | 15.8586 |
| 60 | False | proxy | normal | 7 | 33.4241 | 10.0452 |
| 60 | False | proxy | pred14 | 7 | 35.8527 | 10.0452 |
| 60 | False | proxy | oracle14 | 7 | 22.6045 | 10.0452 |
| 60 | False | proxy | oracle_supported14 | 7 | 17.7323 | 10.0452 |
| 60 | True | real | normal | 16 | 37.5776 | 23.6068 |
| 60 | True | real | pred14 | 16 | 37.8957 | 23.6068 |
| 60 | True | real | oracle14 | 16 | 33.1447 | 23.6068 |
| 60 | True | real | oracle_supported14 | 16 | 28.9090 | 23.6068 |
| 60 | True | proxy | normal | 11 | 37.4540 | 25.9564 |
| 60 | True | proxy | pred14 | 11 | 38.7549 | 25.9564 |
| 60 | True | proxy | oracle14 | 11 | 36.3402 | 25.9564 |
| 60 | True | proxy | oracle_supported14 | 11 | 31.7312 | 25.9564 |
