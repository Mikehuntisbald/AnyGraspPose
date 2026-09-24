# 多尺度局部差分 JEPA V15

从 V14 step11000 接续至12000，新增1000次优化器更新。source SHA256: `49d49a51d58ec566000a9050465b7304989a2d5be74ad78f3d76bc6aae1166ac`。只训练JEPA恢复及在线DINO；历史关闭、位姿/query/readout冻结；4/11层输入与监督各0.625。保留完整模型、EMA1250、8rank RNG和sampler352000。目标函数变化后重建AdamW及1000步scheduler，warmup25步，其他学习率不变。

新增多尺度局部差分（权重0.2）：1/2/4patch距离，水平/垂直/双对角；按有效样本和有效尺度归一化，teacher差分RMS下限0.1，SmoothL1 beta0.1。作用于逐patch LayerNorm后的特征，空间共同加性分量在这个空间消除，不宣称完全去语义。

原全局中心化从0.25降至0.05；保留全局软对应0.05；新增5x5邻域残差软对应0.05（温度0.1）。局部均值只读同来源有效位置；teacher相似位置保留软歧义。CAD表面对应0.1、绝对特征和XYZ/depth/一致性权重不变。初版保留原绝对特征权重，不同时扫权重。

真实与CAD代理分别计算。只有缺失query接收梯度，非query端点替换为同源、停止梯度teacher参考（仅loss内部）；长距离边沿途所有patch必须有效；没有跨真实/CAD拼接边的差分，没有预测置信度屏蔽。无新增推理模块、无GT输入student、无可见预测恢复梯度。

评估使用独立的固定step11000 EMA teacher，固定40条序列/原crop协议。初始11000、11500、12000记录两层的局部差分误差/相对RMSE/幅度比/有效配对质量，空间检索、CAD对应和物理几何误差。训练目标仍是在线EMA；不把移动teacher的训练loss下降当作几何改进。不运行official test、native pose gate、多seed或预算扩展。

每50步保存完整断点，11002→11003跨进程恢复和全状态审计，CPU约束测试及H20完整40帧前后向预检。固定源码运行目录 `/mnt/why/dexycb_lip/unified_jepa_20260921/local_difference_v15`；controller负责串行独占8卡训练/评估。
