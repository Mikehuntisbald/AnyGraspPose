# V17 RoPE 开／关：当前权重几乎没有几何收益

结论：40序列、8000帧配对，重遮挡XYZ/深度差异约0.0002–0.0011mm，法线平均差异不足0.001°；相关重遮挡几何指标的配对区间均跨0。关闭RoPE不能修复当前XYZ退化。轻遮挡CAD匹配交叉熵略降，但未转化为可用的几何收益。

开启状态的363个汇总值逐值复现上一轮终止评估；8rank模型全部张量在评估后保持不变，checkpoint文件SHA256未变。此次只是同权重推理干预，不能推断移除RoPE后重新训练的结果。

Checkpoint step24400, SHA256 `ec0732f609fd04dc7aee50a42c1ae7b9830a9e9acf8f1ac9a614efc45e50add5`.

Fixed40, identical encoded observation/teacher per pair. Off changes only four RoPE gains to0; all model tensors restored and audited. No optimizer updates. History disabled and pose weights frozen.

Differences below are on minus off. Geometry/angle/cross-entropy: negative favors RoPE. Retrieval/support: positive favors RoPE. This cannot establish whether retraining without RoPE would be better.

| Case | Target / metric | On | Off | On - off | 95% paired CI |
|---|---|---:|---:|---:|---|
| natural | geometry_focus_proxy/xyz_mm | 54.552401 | 54.554028 | -0.001628 | [-0.003900, +0.000487] |
| natural | geometry_focus_proxy/depth_mm | 15.923404 | 15.921151 | +0.002253 | [-0.000048, +0.004672] |
| natural | normal_proxy/angle_deg | 80.829103 | 80.855298 | -0.026196 | [-0.083357, +0.016115] |
| natural | spatial_cad_proxy/retrieval_top1 | 0.408836 | 0.408373 | +0.000463 | [+0.000000, +0.001389] |
| natural | cad_match_proxy/cross_entropy | 4.528756 | 4.530919 | -0.002163 | [-0.005452, +0.000128] |
| natural | cad_match_proxy/top1_supported | 0.305642 | 0.304157 | +0.001486 | [-0.007283, +0.012269] |
| light | geometry_focus_real/xyz_mm | 57.223902 | 57.224795 | -0.000893 | [-0.002805, +0.001090] |
| light | geometry_focus_real/depth_mm | 10.253250 | 10.252700 | +0.000549 | [-0.001463, +0.002736] |
| light | normal_real/angle_deg | 71.245771 | 71.260774 | -0.015003 | [-0.036938, +0.006025] |
| light | geometry_focus_proxy/xyz_mm | 63.304097 | 63.303880 | +0.000218 | [-0.003090, +0.003619] |
| light | geometry_focus_proxy/depth_mm | 18.653318 | 18.650906 | +0.002412 | [-0.000973, +0.007027] |
| light | normal_proxy/angle_deg | 81.182984 | 81.185159 | -0.002175 | [-0.060338, +0.043540] |
| light | spatial_hidden_real/retrieval_top1 | 0.308877 | 0.308877 | +0.000000 | [+0.000000, +0.000000] |
| light | spatial_cad_proxy/retrieval_top1 | 0.279378 | 0.279378 | +0.000000 | [+0.000000, +0.000000] |
| light | cad_match_real/cross_entropy | 3.905205 | 3.906998 | -0.001793 | [-0.003241, -0.000587] |
| light | cad_match_proxy/cross_entropy | 5.602988 | 5.608594 | -0.005606 | [-0.014217, -0.000026] |
| light | cad_match_real/top1_supported | 0.418046 | 0.418202 | -0.000156 | [-0.004858, +0.004354] |
| light | cad_match_proxy/top1_supported | 0.252102 | 0.251456 | +0.000646 | [-0.001132, +0.002813] |
| heavy_pooled | geometry_focus_real/xyz_mm | 60.562966 | 60.562633 | +0.000333 | [-0.000005, +0.000800] |
| heavy_pooled | geometry_focus_real/depth_mm | 13.429843 | 13.430246 | -0.000403 | [-0.001072, +0.000257] |
| heavy_pooled | normal_real/angle_deg | 72.072967 | 72.073656 | -0.000689 | [-0.009155, +0.007300] |
| heavy_pooled | geometry_focus_proxy/xyz_mm | 55.908477 | 55.908309 | +0.000168 | [-0.001006, +0.001315] |
| heavy_pooled | geometry_focus_proxy/depth_mm | 19.714616 | 19.713557 | +0.001058 | [-0.002938, +0.004386] |
| heavy_pooled | normal_proxy/angle_deg | 78.828470 | 78.829360 | -0.000891 | [-0.088271, +0.079047] |
| heavy_pooled | spatial_hidden_real/retrieval_top1 | 0.408050 | 0.408386 | -0.000335 | [-0.001014, +0.000312] |
| heavy_pooled | spatial_cad_proxy/retrieval_top1 | 0.309430 | 0.309207 | +0.000223 | [-0.000298, +0.000966] |
| heavy_pooled | cad_match_real/cross_entropy | 4.241477 | 4.241670 | -0.000193 | [-0.000674, +0.000171] |
| heavy_pooled | cad_match_proxy/cross_entropy | 4.630322 | 4.631079 | -0.000758 | [-0.001903, +0.000162] |
| heavy_pooled | cad_match_real/top1_supported | 0.360489 | 0.360529 | -0.000040 | [-0.001391, +0.001191] |
| heavy_pooled | cad_match_proxy/top1_supported | 0.284063 | 0.287033 | -0.002971 | [-0.008534, +0.000171] |


[验证回执](receipt.json) · [完整指标](summary.json) · [配对序列指标](paired_sequence_metrics.csv) · [图表](rope_switch.png)
