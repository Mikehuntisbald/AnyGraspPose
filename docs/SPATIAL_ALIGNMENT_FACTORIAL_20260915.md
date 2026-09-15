# 显式空间对齐监督 × 父 latent：2×2 验证协议

启动时状态：实现、测试、真实接口、显存和 DDP 恢复预检完成；四组
固定预算训练已在八张 H20 上并行启动。**目前没有这轮最终精度，
不能把 pilot、训练 loss 或非零梯度当成验证收益。**

最终结果由完整评测后的 receipt 和归档决定。自动回传状态位于
`runs/spatial_alignment_factorial_20260915_completed/status.json`；完成后
生成 `docs/SPATIAL_ALIGNMENT_FACTORIAL_RESULTS_20260915.md`。

## 四组严格保持的条件

| 组名 | 显式空间监督 | 新分支直接读取父 latent | GPU |
|---|---|---|---|
| s0_l1 | 关闭 | 保留 | 0,1 |
| s0_l0 | 关闭 | 切断 | 2,3 |
| s1_l1 | 开启，权重 0.05 | 保留 | 4,5 |
| s1_l0 | 开启，权重 0.05 | 切断 | 6,7 |

四组来自相同的 residual 父权重和相同随机初始化的新分支，新增
旋转输出从零开始。不是从已训练 alignment final1000 或诊断短拟合
权重继续。固定父权重 SHA：
`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`。

各 final1000、seed42、相同 64,000 次片段采样，effective batch64。
每组两卡、每卡 batch16、累积两次；原 8+48 观察窗口、48 个监督
位置，前八帧监督和 S1O1 遮挡配方保留。50% 确定性真实 PoseCNN
初始化请求，缺失时沿用已记录的训练回退；RGB 冻结，其他父模块
联合适配。fresh AdamW，加载模块 LR2e−6、其他原新模块 1e−5、
旋转新分支 1e−4，warmup100 / cosine 至 0.1 倍。没有改变数据预算。

L0 将新分支 MLP 输入中的父 latent 替换为零，保留相同参数形状。
这切断的是新增分支的直接 latent 路径；原父模型仍正常运行，dense
观测特征也已经融合过几何，**不能称“完全无父模型/无几何/无时序”。**

## 空间监督的具体定义

监督实际 `object-query ← CAD-key` cross-attention 的平均头概率：

1. 从 base pose 渲染的深度/轮廓/物体 XYZ，取 14×14 单元中心的
   CAD 表面代表点。米制、mesh 中心和相机系保持现有定义。
2. 当前预测产生之后，训练端才用当前 GT pose 把这些物体点投影到
   同一 crop，双线性分配到邻近 query 单元，生成多正例目标分布。
3. 排除出界/背后、GT CAD 自遮挡、无效观测深度及前景遮挡/深度
   不一致点。自遮挡容差 `max(2 mm,0.01d)`；观测一致性容差
   `max(5 mm,0.02d)`。读取增强后的传感器深度，不读取手或物体
   segmentation 标签。
4. 用实际 attention 的 Q/K 投影计算 FP32 交叉熵，平均到有支持的
   query；原位姿 loss 不变，再加 `0.05 × L_spatial`。没有支持时
   辅助项为零，仍保留完整位姿监督；不会生成伪对应或 NaN。

模型常规 SDPA 前向保持原路径。额外 Q/K 概率计算和 GT 渲染只用于
训练 loss，推理不生成 GT 目标，也不额外渲染 GT。辅助项能从零
rotation head 起点直接给 CAD/匹配模块梯度。

官方缓存 `bop/models/models_info.json` 中有离散/连续对称元数据的
ID 为 **1,13,16,18,19,20,21**，本版排除它们的辅助点对应监督，
避免强迫几何不可辨别的身份对应；全部仍参与原位姿 loss、完整 val
和主指标。这是辅助项的适用范围，不是从评测中删掉难例。

## 已实际通过的检查

- 65 项不同的 CPU 检查：主套件 61、扩展 13，其中 9 项重复。
  包括投影平移/旋转方向、非恒等 crop 内参、深度/自遮挡过滤、
  对称排除、空监督梯度、实际 attention 概率一致性、L0 输出
  不依赖 latent、checkpoint 路径开关防错、factorial 对比符号。
- 4 项 CUDA 检查；真实 RGB-D 四组零起点分别在 CPU FP32 一次更新、
  GPU FP32/BF16 八次更新上与父模型逐位一致。
- 每组真实 3 步 pilot 已完成并丢弃。S1 的 CAD/QK 在首步即有非零
  梯度；例如首步 CAD 梯度范数 0.05854，QK 为 0.12140。
  同一 pilot 中目标命中率由 5.61% 增至 12.16% / 14.37%，这仅为
  训练内连通性证据，支持点集合也会随各自闭环变化。
- 单卡 batch16 真实显存：S0 32.57 GiB、S1 34.47 GiB 的 PyTorch
  peak allocated；不是整卡驱动占用。四组双卡 DDP 3 步保存、恢复
  至第 4 步，optimizer/scheduler/RNG/sampler 检查通过。
- 修正了梯度累积时启动监督帧数只记录最后一个 microbatch 的
  日志问题。当前每 rank 每步启动监督 256 帧，总监督 1,536 帧，
  全局总监督 3,072 帧；辅助日志也汇总两个 microbatch。

## 覆盖率限制：实测而非假设

核查了 16 个预先固定 train 片段的未增强首个更新帧。一个无遮挡
物体 8 样本有 30 个合法 CAD 点，GT 自可见性和传感器深度有效性
均通过，但观测深度比 GT CAD 表面深 **5.20–19.30 mm**，中位
**12.77 mm**，全部超过本次 5 mm 一致性门槛，最终没有辅助点。
其自渲染投影深度残差中位只有 0.248 mm，没有在此例发现投影方向
错误。这里的差异不能未经进一步检查就归因于传感器或 GT 某一方。

因此“无辅助点”不等于“被遮挡”。正式训练保留固定门槛，同时
记录 eligible clips、visible keys、supported queries、CE 与命中率；
完整结果必须连同覆盖率解释。此次不依据已运行曲线修改门槛。

[真实图像与训练目标](../runs/spatial_alignment_factorial_20260915_preflight/targets.png)。
图中青线表示监督目标位移，红点是目标单元，**不是模型预测对应**。
可视化使用 CPU renderer、无额外增强；正式训练使用已验证的 CUDA
renderer 和原 S1O1 增强。

## 正式判定与自动后处理

四组训练完成后，先在当前源码下复现旧父模型完整轨迹，再对每组
运行同一真实 PoseCNN 初始化的 native s0 val：320 流 / 23,200 帧。
所有帧递推、每帧一次校正/历史提交、全程零 FP、无 GT reset，
初始化缺失仍算失败。推理与读取 GT 的计分进程分开。

预设五个 factorial 对比：有/无 latent 时各自的空间监督效应、
有/无空间监督时各自移除 latent 的效应，以及两者 interaction。
主项为坏初值前八帧 canonical 旋转误差；同时检查总体 ADD/严格
ADD-S、严重遮挡、启动中心、长遮挡恢复、对称/非对称分层及延迟。
FP 使用已封存的同协议纯 FP 基线作为外部参考。

固定 40 条物理序列、10,000 次共享 bootstrap，输出逐帧配对 CSV、
物理序列/物体汇总、95% 区间；主旋转五对比另报 99% 区间用于
Bonferroni 五对比族控制。其他多指标/多分层不声称全局校正，也
不把单种子区间当作训练种子稳定性。没有自动晋级或新 official test。

完成后自动运行隔离 batch1 测速、生成完整结果和图表、归档回传并
逐个验 SHA；本地 watcher 只在校验通过后写最终报告。

## 代码、状态和恢复

服务器独立运行目录：`/mnt/why/dexycb_lip/spatial_alignment_2x2_20260915/`。
本地与运行目录的核心源码 SHA 相同：
`87d4591a3fceee004d1a43a099fba958e6115579f283dffada2f06d7aa0fca6f`。
这是源码复制/哈希对应，不表示远端 Git 分支已经 commit/push。

- 实验约定：`runs/factorial/experiment.json`
- 四组训练日志：`runs/factorial/{s0_l1,s0_l0,s1_l1,s1_l0}/train/rank{0,1}.jsonl`
- 控制器状态：`runs/factorial/status.json`
- 训练后评测：`runs/factorial/evaluation/<arm>/scored/`
- 完整条件对比：`runs/factorial/analysis/report.json`
- 终态归档状态：`runs/completion/status.json`
- 本地自动回传：`runs/spatial_alignment_factorial_20260915_completed/status.json`
- 已通过的预检：`runs/spatial_alignment_factorial_20260915_preflight/evidence/`

对应进程已退出且保存了 `last.pt` 时，可按原参数恢复单组，例如：

```bash
cd /mnt/why/dexycb_lip/spatial_alignment_2x2_20260915
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  /mnt/why/dexycb_lip/.venv-fp/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m lip.train_stream \
  --config runs/factorial/s1_l0/config.yaml \
  --output runs/factorial/s1_l0/train \
  --resume runs/factorial/s1_l0/train/last.pt --max-steps 1000 \
  --data-root /mnt/why/dexycb_lip/cache/raw_full_20260910 \
  --index-root /mnt/why/dexycb_lip/cache/dexycb_s0 \
  --fixed-manifest /mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/pair/training_samples.json
```

原始数据、宿主 CUDA/驱动和保留权重均未修改。
