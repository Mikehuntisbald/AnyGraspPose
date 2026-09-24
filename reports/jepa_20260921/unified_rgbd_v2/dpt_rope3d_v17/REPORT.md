# V17 DPT + 3D RoPE + XYZ-derived normals — completed

已完成23400→24400，共1000次新增更新。8rank终止审计通过，终止权重已同步本地并校验SHA256。按预算停止，不自动追加训练。最后50步平均3.167秒/步。

结论：法线方向误差有所下降，但DPT的绝对XYZ恢复尚未追上源MLP；深度仍略差，XYZ/深度一致性也退步。该实验同时修改DPT、RoPE及法线loss，未完成RoPE开关消融，不能单独归因。位姿路径冻结，这不是位姿收益评估。

固定40条验证序列、同一固定step11000 EMA teacher、相同crop协议；下表为受控重遮挡heavy_pooled，history关闭，误差越低越好。真实目标指原本可见后被人工遮挡的区域，CAD代理为原本不可见区域。

| 区域 / 指标 | 源MLP | +500步 | +1000步 |
|---|---:|---:|---:|
| 真实 / XYZ mm | 37.83 | 60.77 | 60.56 |
| 真实 / 深度 mm | 12.13 | 15.93 | 13.43 |
| 真实 / 法线角度° | 88.46 | 83.91 | 72.07 |
| CAD代理 / XYZ mm | 37.16 | 55.24 | 55.91 |
| CAD代理 / 深度 mm | 18.36 | 20.13 | 19.71 |
| CAD代理 / 法线角度° | 88.79 | 83.35 | 78.83 |

真实区域XYZ的恒定物体中心基线为61.69mm，新模型60.56mm仅略好于该基线；这提示绝对表面定位仍很弱，并不直接证明输出恒定。XYZ/深度不一致从源模型19.16mm增至34.84mm。法线改善不能替代绝对几何恢复。

真实隐藏patch检索41.16%→40.81%，CAD代理32.94%→30.94%。最终RoPE门控约0.0070/0.0143/0.0086/0.0091；门控非零不是有效性证据。

共享模型、DINO/EMA、RNG继承；DPT随机初始化、RoPE零门控，AdamW/scheduler重建。新模块1e-4、已有JEPA1e-5、DINO1e-6，50步warmup，历史关闭、位姿冻结。CPU与H20前后向、23402→23403跨进程恢复、终止状态审计均通过。

Terminal SHA256: `ec0732f609fd04dc7aee50a42c1ae7b9830a9e9acf8f1ac9a614efc45e50add5`。

[完整配对结果](comparison/REPORT.md) · [图表](comparison/geometry_normals.png) · [终止审计](final/audit1000.json) · [收集回执](collection_receipt.json) · [结构与损失](../../../../docs/DPT_ROPE3D_NORMALS_V17.md)
