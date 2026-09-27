# V74 dense backward-correspondence audit

128repeated cases; primary geometry and existing sparse-flow records match the original evaluation exactly.
Dense flow maps observation pixels to the estimated CAD raster; it is not the sparse forward flow in the main report.
EPE is measured on fixed GT-supported correspondences, without predicted-gate filtering. Gate fraction uses all region pixels.

| Angle | Heavy | Stage | Region | Frames | GT support | Gate on | Dense EPE px | Zero-flow EPE px |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 0 | False | 0 | real | 8 | 100.0% | 100.0% | 0.2355 | 0.0003 |
| 0 | False | 0 | proxy | 12 | 99.2% | 100.0% | 0.2692 | 0.0003 |
| 0 | False | 1 | real | 8 | 100.0% | 100.0% | 0.2320 | 0.0003 |
| 0 | False | 1 | proxy | 12 | 99.2% | 100.0% | 0.2679 | 0.0003 |
| 0 | True | 0 | real | 16 | 100.0% | 100.0% | 0.2775 | 0.0004 |
| 0 | True | 0 | proxy | 8 | 100.0% | 100.0% | 0.3007 | 0.0008 |
| 0 | True | 1 | real | 16 | 100.0% | 100.0% | 0.2738 | 0.0004 |
| 0 | True | 1 | proxy | 8 | 100.0% | 100.0% | 0.2994 | 0.0008 |
| 10 | False | 0 | real | 16 | 99.9% | 97.1% | 4.7156 | 4.6907 |
| 10 | False | 0 | proxy | 22 | 95.7% | 94.1% | 4.0931 | 4.1776 |
| 10 | False | 1 | real | 16 | 99.9% | 97.1% | 4.7159 | 4.6907 |
| 10 | False | 1 | proxy | 22 | 95.7% | 94.1% | 4.0928 | 4.1776 |
| 10 | True | 0 | real | 32 | 99.1% | 97.5% | 5.0516 | 5.0544 |
| 10 | True | 0 | proxy | 16 | 98.1% | 96.0% | 4.3159 | 4.2447 |
| 10 | True | 1 | real | 32 | 99.1% | 97.5% | 5.0509 | 5.0544 |
| 10 | True | 1 | proxy | 16 | 98.1% | 96.0% | 4.3107 | 4.2447 |
| 60 | False | 0 | real | 8 | 84.0% | 83.5% | 35.1335 | 35.1533 |
| 60 | False | 0 | proxy | 12 | 62.0% | 61.7% | 31.1399 | 31.4111 |
| 60 | False | 1 | real | 8 | 84.0% | 83.5% | 35.1347 | 35.1533 |
| 60 | False | 1 | proxy | 12 | 62.0% | 61.6% | 31.1435 | 31.4111 |
| 60 | True | 0 | real | 15 | 86.1% | 82.9% | 26.7851 | 26.8943 |
| 60 | True | 0 | proxy | 8 | 74.5% | 65.7% | 23.4254 | 23.4647 |
| 60 | True | 1 | real | 15 | 86.1% | 82.9% | 26.7867 | 26.8943 |
| 60 | True | 1 | proxy | 8 | 74.5% | 65.7% | 23.4231 | 23.4647 |
