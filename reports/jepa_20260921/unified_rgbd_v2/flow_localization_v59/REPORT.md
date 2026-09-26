# V59 frozen flow localization audit

256 controlled training-partition physical-holdout frames. Frozen V56 weights; no training or pose evaluation.
All means below weight each eligible frame equally. Heavy denotes requested augmentation, not a measured visibility threshold.
Oracle box EPE is an unattainable-or-equal diagnostic lower bound for the existing ±14px endpoint correction; it is not a model result.

| Case | Frames / points | Zero EPE | Round 0 EPE | Round 1 coarse | Round 1 EPE | No feedback EPE | Outside correction box | Oracle box EPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| controlled_10/heavy/observed | 58 / 228 | 5.978 | 7.818 | 5.985 | 5.950 | 7.853 | 3.3% | 0.145 |
| controlled_10/heavy/proxy | 27 / 143 | 4.450 | 8.063 | 6.201 | 6.178 | 8.070 | 4.5% | 0.448 |
| controlled_10/heavy/real | 62 / 982 | 4.976 | 7.930 | 5.778 | 5.766 | 7.896 | 2.3% | 0.076 |
| controlled_10/nonheavy/observed | 61 / 1227 | 5.594 | 7.134 | 5.661 | 5.641 | 7.126 | 2.2% | 0.077 |
| controlled_10/nonheavy/proxy | 26 / 122 | 3.706 | 8.751 | 6.672 | 6.690 | 8.746 | 6.8% | 0.216 |
| controlled_10/nonheavy/real | 26 / 145 | 6.080 | 9.104 | 6.522 | 6.502 | 8.985 | 6.4% | 0.225 |
| controlled_60/heavy/observed | 54 / 294 | 33.389 | 27.854 | 26.982 | 27.015 | 27.826 | 63.1% | 13.754 |
| controlled_60/heavy/proxy | 32 / 123 | 27.979 | 27.182 | 26.079 | 26.103 | 26.837 | 74.2% | 12.484 |
| controlled_60/heavy/real | 59 / 629 | 30.055 | 28.134 | 28.565 | 28.623 | 28.186 | 71.2% | 15.114 |
| controlled_60/nonheavy/observed | 58 / 754 | 28.993 | 22.664 | 20.885 | 20.927 | 22.679 | 58.9% | 8.319 |
| controlled_60/nonheavy/proxy | 32 / 149 | 29.093 | 26.916 | 24.397 | 24.399 | 26.982 | 62.6% | 11.220 |
| controlled_60/nonheavy/real | 24 / 93 | 25.414 | 22.517 | 20.442 | 20.478 | 22.383 | 63.7% | 7.121 |
