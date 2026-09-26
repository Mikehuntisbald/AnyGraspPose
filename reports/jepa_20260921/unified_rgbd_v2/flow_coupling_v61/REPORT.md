# V61 flow write dependence

64 frozen V60-extra observations, controlled10/60 degree references, training-partition physical holdout.
Oracle variants replace only known supported front-surface endpoints; other predictions are unchanged. These are diagnostics, not deployable models.
Heavy is the requested augmentation condition. Means weight eligible frames equally; no predicted confidence filters the targets.

| Angle | Variant | Source | Heavy frames | XYZ mm | Depth mm | Output XYZ change mm |
|---:|---|---|---:|---:|---:|---:|
| 10 | normal | real | 15 | 9.8538 | 7.0013 | 0.0000 |
| 10 | normal | proxy | 10 | 7.8585 | 5.4630 | 0.0000 |
| 10 | off | real | 15 | 9.8559 | 7.0018 | 0.0860 |
| 10 | off | proxy | 10 | 7.8583 | 5.4718 | 0.0841 |
| 10 | gain10 | real | 15 | 9.8513 | 6.9971 | 0.1577 |
| 10 | gain10 | proxy | 10 | 7.8495 | 5.4700 | 0.1511 |
| 10 | gain100 | real | 15 | 9.8483 | 6.9916 | 1.1955 |
| 10 | gain100 | proxy | 10 | 7.8173 | 5.4373 | 1.0682 |
| 10 | oracle | real | 15 | 9.8512 | 7.0018 | 0.0777 |
| 10 | oracle | proxy | 10 | 7.8670 | 5.4670 | 0.0904 |
| 10 | oracle_gain10 | real | 15 | 9.8473 | 6.9979 | 0.1489 |
| 10 | oracle_gain10 | proxy | 10 | 7.8403 | 5.4743 | 0.1586 |
| 60 | normal | real | 16 | 37.9733 | 17.1331 | 0.0000 |
| 60 | normal | proxy | 8 | 49.3692 | 8.5497 | 0.0000 |
| 60 | off | real | 16 | 37.9770 | 17.1329 | 0.1001 |
| 60 | off | proxy | 8 | 49.3665 | 8.5406 | 0.0554 |
| 60 | gain10 | real | 16 | 37.9531 | 17.1294 | 0.1578 |
| 60 | gain10 | proxy | 8 | 49.2847 | 8.5348 | 0.2481 |
| 60 | gain100 | real | 16 | 37.7378 | 17.1159 | 1.2368 |
| 60 | gain100 | proxy | 8 | 49.1566 | 8.4099 | 0.8465 |
| 60 | oracle | real | 16 | 37.9713 | 17.1342 | 0.0699 |
| 60 | oracle | proxy | 8 | 49.3198 | 8.5435 | 0.1911 |
| 60 | oracle_gain10 | real | 16 | 37.9565 | 17.1275 | 0.1619 |
| 60 | oracle_gain10 | proxy | 8 | 49.3026 | 8.5237 | 0.2010 |

Write/patch norm ratios: 0.067672%, 0.068709%.
Patch addition dtypes: [['torch.float32'], ['torch.float32']].
Oracle correspondence coverage: {'frames': 64, 'zero_correctable_frames': 1, 'minimum_points': 0, 'mean_points': 19.578125}.

Normal rewrite is bitwise identical to production output on every tested frame. Off/gain/oracle modes are frozen interventions; no training was performed.
