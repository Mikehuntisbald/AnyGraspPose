# V17 DPT + 3D RoPE + XYZ-derived normals

已从完整step23400初始化，新增1000步，目标step24400。旧V16已在完整断点停止，8rank没有丢失已完成更新。只训练JEPA恢复；pose/query/relation冻结，history关闭；保留在线DINO与EMA及原4/11等权特征目标。

DPT读取四层JEPA输出，经64/32/16/8金字塔和残差融合恢复224×224 XYZ、深度残差与有效性，替换原MLP。CAD SurfaceRead启用测量XYZ的3D RoPE。法线由预测XYZ的1/2像素有限差分导出，使用0.1×(真实区域损失+0.5×CAD代理损失)，不新增法线头。所有stencil必须留在同一有效监督来源内；不监督未人工遮挡的可见预测区域。

共享模型权重、EMA counter13650、sampler和8rank RNG继承；DPT随机初始化、RoPE零门控。新建AdamW与50步warmup；新模块1e-4、已有JEPA1e-5、在线DINO1e-6。旧MLP不在新模型中，不能期待初始化时几何输出与旧模型一致。

验证：29项CPU结构/迁移/梯度测试通过；完善宽stencil边界后，20项法线/局部loss回归通过。H20真实batch4×40帧预检通过，所有新模块/共享主干都有有效梯度，teacher/crop参考不变，冻结路径无梯度。热身后2.825秒/episode（单卡前后向，不含8卡通信/优化器），首次编译约315秒，峰值显存9.74GB。这不是正式训练速度或质量收益结论。

23402→23403跨进程恢复及启动审计已通过：8rank均完成3次新增更新，共享初始权重和冻结位姿逐值一致，新DPT与RoPE已更新。控制器正在执行至23900（+500）；随后在+500/+1000做fixed40评估，最终补跑同协议源MLP对照。评估用固定step11000 EMA特征teacher，输出真实/CAD的XYZ、深度、法线角误差、局部检索/对应及法线图。三个改动联合测试，不单独归因收益；不训练位姿、不扩大预算。

[结构与损失说明](../../../../docs/DPT_ROPE3D_NORMALS_V17.md) · [实时状态](LIVE_STATUS.md) · [H20预检](preflight_receipt.json)
