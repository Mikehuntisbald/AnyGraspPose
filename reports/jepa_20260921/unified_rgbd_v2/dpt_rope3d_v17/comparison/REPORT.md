# DPT + 3D RoPE + XYZ-derived normal supervision

Same fixed40 protocol and fixed step11000 EMA feature teacher. Pose is frozen; this is recovery evaluation. Three changes are combined, so differences cannot identify their individual effects.

| Case | Target / metric | Source MLP | +500 | +1000 |
|---|---|---:|---:|---:|
| natural | geometry_focus_proxy/xyz_mm | 36.37353 | 54.10963 | 54.55240 |
| natural | geometry_focus_proxy/depth_mm | 14.73456 | 17.52524 | 15.92340 |
| natural | normal_proxy/angle_deg | 89.09786 | 84.09842 | 80.82910 |
| natural | spatial_cad_proxy/retrieval_top1 | 0.39569 | 0.41567 | 0.40884 |
| light | geometry_focus_real/xyz_mm | 31.89615 | 57.60043 | 57.22390 |
| light | geometry_focus_real/depth_mm | 8.75199 | 12.57228 | 10.25325 |
| light | normal_real/angle_deg | 88.71013 | 89.48718 | 71.24577 |
| light | geometry_focus_proxy/xyz_mm | 40.04289 | 63.44991 | 63.30410 |
| light | geometry_focus_proxy/depth_mm | 17.29712 | 20.20071 | 18.65332 |
| light | normal_proxy/angle_deg | 89.44827 | 75.57029 | 81.18298 |
| light | spatial_hidden_real/retrieval_top1 | 0.33650 | 0.30571 | 0.30888 |
| light | spatial_cad_proxy/retrieval_top1 | 0.27800 | 0.30154 | 0.27938 |
| heavy_pooled | geometry_focus_real/xyz_mm | 37.83146 | 60.77498 | 60.56297 |
| heavy_pooled | geometry_focus_real/depth_mm | 12.12840 | 15.92917 | 13.42984 |
| heavy_pooled | normal_real/angle_deg | 88.46255 | 83.91285 | 72.07297 |
| heavy_pooled | geometry_focus_proxy/xyz_mm | 37.16006 | 55.24254 | 55.90848 |
| heavy_pooled | geometry_focus_proxy/depth_mm | 18.36482 | 20.12762 | 19.71462 |
| heavy_pooled | normal_proxy/angle_deg | 88.79023 | 83.34953 | 78.82847 |
| heavy_pooled | spatial_hidden_real/retrieval_top1 | 0.41160 | 0.40756 | 0.40805 |
| heavy_pooled | spatial_cad_proxy/retrieval_top1 | 0.32941 | 0.30554 | 0.30943 |
