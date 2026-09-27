# Canonical prior quality and atlas selection diagnostic

Frozen V60-extra;64physical-holdout frames inside the training partition. GT prior/endpoints are diagnostic only.
Primary errors include all original target pixels. Covered-only errors are secondary diagnostics, not replacement metrics.
Coverage is mean per-frame target coverage; prior/final errors are means over eligible frames. No pose/depth improvement claim.

| Angle | Heavy | Region | Variant | Frames | Coverage | Prior XYZ mm | Final XYZ mm | Covered prior mm | Covered final mm |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 10 | False | observed | normal | 16 | 0.0% | 12.360 | 12.993 | n/a | n/a |
| 10 | False | observed | local_projection | 16 | 0.0% | 12.360 | 11.914 | n/a | n/a |
| 10 | False | observed | pred14 | 16 | 99.1% | 14.971 | 14.342 | 14.856 | 14.206 |
| 10 | False | observed | oracle_supported14 | 16 | 93.8% | 13.543 | 13.017 | 13.398 | 12.882 |
| 10 | False | observed | perfect_prior | 16 | 100.0% | 0.000 | 2.554 | 0.000 | 2.554 |
| 10 | False | real | normal | 8 | 0.0% | 15.834 | 15.364 | n/a | n/a |
| 10 | False | real | local_projection | 8 | 0.0% | 15.834 | 15.055 | n/a | n/a |
| 10 | False | real | pred14 | 8 | 99.6% | 16.264 | 15.993 | 16.178 | 15.904 |
| 10 | False | real | oracle_supported14 | 8 | 100.0% | 13.169 | 12.870 | 13.160 | 12.862 |
| 10 | False | real | perfect_prior | 8 | 100.0% | 0.000 | 2.718 | 0.000 | 2.718 |
| 10 | False | proxy | normal | 12 | 0.0% | 10.616 | 11.360 | n/a | n/a |
| 10 | False | proxy | local_projection | 12 | 0.0% | 10.616 | 9.922 | n/a | n/a |
| 10 | False | proxy | pred14 | 12 | 100.0% | 12.080 | 11.426 | 12.080 | 11.426 |
| 10 | False | proxy | oracle_supported14 | 12 | 84.7% | 12.547 | 11.849 | 12.384 | 11.911 |
| 10 | False | proxy | perfect_prior | 12 | 100.0% | 0.000 | 2.324 | 0.000 | 2.324 |
| 10 | True | observed | normal | 16 | 0.0% | 11.942 | 11.260 | n/a | n/a |
| 10 | True | observed | local_projection | 16 | 0.0% | 11.942 | 11.520 | n/a | n/a |
| 10 | True | observed | pred14 | 16 | 98.9% | 14.599 | 13.891 | 14.344 | 13.629 |
| 10 | True | observed | oracle_supported14 | 16 | 89.2% | 11.576 | 11.162 | 12.190 | 11.860 |
| 10 | True | observed | perfect_prior | 16 | 100.0% | 0.000 | 2.325 | 0.000 | 2.325 |
| 10 | True | real | normal | 16 | 0.0% | 12.991 | 13.458 | n/a | n/a |
| 10 | True | real | local_projection | 16 | 0.0% | 12.991 | 12.720 | n/a | n/a |
| 10 | True | real | pred14 | 16 | 99.4% | 15.381 | 14.809 | 14.979 | 14.344 |
| 10 | True | real | oracle_supported14 | 16 | 98.3% | 11.899 | 11.544 | 10.927 | 10.460 |
| 10 | True | real | perfect_prior | 16 | 100.0% | 0.000 | 2.301 | 0.000 | 2.301 |
| 10 | True | proxy | normal | 6 | 0.0% | 19.324 | 15.140 | n/a | n/a |
| 10 | True | proxy | local_projection | 6 | 0.0% | 19.324 | 18.073 | n/a | n/a |
| 10 | True | proxy | pred14 | 6 | 83.3% | 20.566 | 19.182 | 18.721 | 17.896 |
| 10 | True | proxy | oracle_supported14 | 6 | 69.4% | 24.873 | 23.992 | 43.183 | 43.182 |
| 10 | True | proxy | perfect_prior | 6 | 100.0% | 0.000 | 2.583 | 0.000 | 2.583 |
| 60 | False | observed | normal | 16 | 0.0% | 33.401 | 33.910 | n/a | n/a |
| 60 | False | observed | local_projection | 16 | 0.0% | 33.401 | 33.281 | n/a | n/a |
| 60 | False | observed | pred14 | 16 | 89.4% | 45.915 | 45.474 | 48.571 | 48.178 |
| 60 | False | observed | oracle_supported14 | 16 | 79.7% | 16.617 | 16.591 | 11.967 | 11.953 |
| 60 | False | observed | perfect_prior | 16 | 100.0% | 0.000 | 2.811 | 0.000 | 2.811 |
| 60 | False | real | normal | 7 | 0.0% | 26.173 | 27.044 | n/a | n/a |
| 60 | False | real | local_projection | 7 | 0.0% | 26.173 | 26.275 | n/a | n/a |
| 60 | False | real | pred14 | 7 | 98.8% | 33.615 | 32.965 | 33.768 | 33.132 |
| 60 | False | real | oracle_supported14 | 7 | 99.0% | 9.871 | 9.497 | 9.662 | 9.291 |
| 60 | False | real | perfect_prior | 7 | 100.0% | 0.000 | 2.956 | 0.000 | 2.956 |
| 60 | False | proxy | normal | 7 | 0.0% | 33.347 | 33.424 | n/a | n/a |
| 60 | False | proxy | local_projection | 7 | 0.0% | 33.347 | 33.309 | n/a | n/a |
| 60 | False | proxy | pred14 | 7 | 78.3% | 39.303 | 38.508 | 43.186 | 42.665 |
| 60 | False | proxy | oracle_supported14 | 7 | 99.9% | 9.595 | 9.588 | 9.579 | 9.572 |
| 60 | False | proxy | perfect_prior | 7 | 100.0% | 0.000 | 2.584 | 0.000 | 2.584 |
| 60 | True | observed | normal | 16 | 0.0% | 32.407 | 33.918 | n/a | n/a |
| 60 | True | observed | local_projection | 16 | 0.0% | 32.407 | 32.793 | n/a | n/a |
| 60 | True | observed | pred14 | 16 | 92.2% | 38.860 | 38.359 | 39.823 | 39.344 |
| 60 | True | observed | oracle_supported14 | 16 | 83.1% | 17.973 | 17.788 | 11.797 | 11.270 |
| 60 | True | observed | perfect_prior | 16 | 100.0% | 0.000 | 2.484 | 0.000 | 2.484 |
| 60 | True | real | normal | 16 | 0.0% | 36.741 | 37.578 | n/a | n/a |
| 60 | True | real | local_projection | 16 | 0.0% | 36.741 | 36.968 | n/a | n/a |
| 60 | True | real | pred14 | 16 | 99.2% | 39.766 | 39.361 | 39.658 | 39.331 |
| 60 | True | real | oracle_supported14 | 16 | 79.2% | 19.483 | 19.803 | 11.349 | 11.483 |
| 60 | True | real | perfect_prior | 16 | 100.0% | 0.000 | 2.569 | 0.000 | 2.569 |
| 60 | True | proxy | normal | 11 | 0.0% | 35.324 | 37.454 | n/a | n/a |
| 60 | True | proxy | local_projection | 11 | 0.0% | 35.324 | 36.096 | n/a | n/a |
| 60 | True | proxy | pred14 | 11 | 98.9% | 42.641 | 42.276 | 42.775 | 42.440 |
| 60 | True | proxy | oracle_supported14 | 11 | 74.7% | 25.901 | 26.749 | 12.929 | 13.172 |
| 60 | True | proxy | perfect_prior | 11 | 100.0% | 0.000 | 2.549 | 0.000 | 2.549 |
