# V16执行加速

V15模型、损失、数据、batch32、40帧episode、4/11等权、历史关闭、冻结位姿路径和25000步学习率调度不变。

- 冻结crop参考DINO及EMA teacher使用原eager算子的CUDA Graph；在线student/反传不换实现，未采用SDPA替换。
- 在后台pin_memory预取开始前捕获全部静态形状；每次replay读取当前EMA权重并复制输出，避免缓存旧teacher目标或覆盖仍使用的输出。
- 缓存像素网格、无CPU同步相机求逆、同CAD多视角批量纹理渲染、teacher图像几何目标批量化、观测几何通道/pool批量化、冻结CAD均值缓存。
- 保持原逐视角矩阵运算、标量除法与逐帧有效样本归一化。frame_batch仍8；16/32未显著提速，未部署。

同输入8卡零更新基准3.756→2.875秒/步，吞吐1.3066倍；全局梯度相对误差0、crop逐值一致、检查的teacher张量最大误差0。CPU23项检查及真实22252→22253跨进程续训通过；模型、AdamW、scheduler、8rank RNG精确继承，EMA连续。已完成更新从22250完整断点接续，无新增训练预算。

正式训练step22276–22295平均3.048秒/步；之前不同窗口约4.119秒/步。GPU平均利用率不同20秒采样36.49%→46.61%，并未满载；这些生产采样不是同输入配对实验。

证据：[匹配基准](../reports/jepa_20260921/unified_rgbd_v2/execution_speed_v16/benchmark.json)、[续训审计](../reports/jepa_20260921/unified_rgbd_v2/execution_speed_v16/startup_receipt.json)、[图表](../reports/jepa_20260921/unified_rgbd_v2/execution_speed_v16/speed_comparison.png)。运行路径和checkpoint为外部实验产物，不随源码推送。
