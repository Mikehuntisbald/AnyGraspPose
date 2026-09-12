# Object ← Context Cross-Attention + Gated Residual

已实现独立架构 `stream_dual_cross`，从旧 dual **8,800 步**迁移并启动八卡正式训练。北京时间 2026-09-12 13:09:43 启动，13:12:42 已核实新 stage 的 **100 步 checkpoint**及完整恢复状态，训练继续运行。实现与训练预检通过；本分支尚无完整 s0 val 精度结果。

## 实际计算

沿用现有两个 readout latent：每帧一个 object token `O_t`，一个 full-context token `C_t`，均为 256 维。object token 是原始 object-biased readout；context token 来自完整 crop，不是手标签或背景专用表示。

```text
原 4 层 source-only temporal Transformer
             │                   │
             O_t                 C_t
             │                   │
             Q             追加独立 context K/V
             │                   │
             └── CrossAttention ─┘  最近 8 帧，包含当前帧
                      │
                      H_t
                      │
        g_t = sigmoid(MLP([O_t, H_t]))
                      │
                O'_t = O_t + g_t H_t
                      │
              原解耦 pose head
```

Cross-attention 使用 256 维、8 heads、dropout=0；gate 为 `512→256→1` MLP，输出标量并广播到 256 维。当前 `O_t`、`H_t` 各只有一个 token，因此 `Pool` 是恒等操作。context 使用最近八帧的读出 latent，不是重算历史 RGB-D，也不是把 context 空间 token 扩展为新的 backbone。

新增 context K/V 与原四层、每帧 17 个 source token 的 KV 完全分开。它只保存当时的 `C_t` 投影，不保存 `O'_t`，也不向 source memory 写入 readout。mask 同时检查 frame window、时间戳、stream tag 和有效位，保证只访问当前或过去的同流 context。仅一帧时只有一个合法 context；历史增长后才具有多 key 的选择能力。

新缓存契约为 `lip-v2-source17-context-readout-cross-gated-v1`。旧架构仍使用原契约，不能互相复用在线状态。初始化/重定位清空两类 cache；小幅外部 pose 修正保留历史表示；非有限 proposal 不提交。训练 burn-in 边界同时 detach source/context KV，监督 unroll 内保留二者梯度；optimizer step 之间不保留 cache。

## 迁移与保留

旧 dual 在 8,800 步原子 checkpoint 完成后保留独立硬链接，核实 scheduler、optimizer 和八个 rank 的 RNG，再停止该训练 launcher。

- 旧 dual 归档：`/mnt/why/dexycb_lip/stream_v2/runs/stream_v2_dual_archive/checkpoint_8800.pt`
- 旧权重 SHA256：`0d9fb0a42c6b1a3adb2673b964450b3d8bd2e900f92074d4ea594cdceb1111ce`
- 新初始化权重：`runs/stream_v2_cross_from_dual_8800_init/init.pt`
- 初始化 SHA256：`ea2c7990ed7b3ec335685bf6a0c9ac1d70af71427ace1e44d9f1846f9b8418f0`
- 已迁移参数：**15,648,398 / 16,043,151，97.5394%**。

RGB/geometry encoder、spatial fusion、temporal Transformer、state/position 模块、两个 readout query 以及 pose head 保留。旧 `context_projection` 被移除；旧 gate 输入 `[O,C]` 改为 `[O,H]`，故 gate 显式重置。新增 attention Q/K/V 随机初始化，输出投影为零，gate 初始 bias=-2，sigmoid 约 0.119。

初始化时 `H=0`，模型输出原始 `O`。旧 dual 的线性 context 残差已被替换，因此初始化后的预测可能改变；不声称这是保持旧输出不变的重参数化。新 stage 从 0 开始，optimizer/scheduler 重新创建；旧 dual 的完整 optimizer/RNG 仍在归档中，可恢复旧训练。

运行目录独立为 `/mnt/why/dexycb_lip/stream_v2_cross_dual_8800`。源码 hash：`7fc30d6c2ff7781ab45cfb5339e9dff631338455a4f57b838d636ec234551095`。旧 dual 运行目录和此前单/双 readout 评估结果保留；未提交或推送 Git，没有上传 checkpoint，没有修改原始数据、驱动或全局 CUDA。

## 真实验证

共 **100 个不同测试通过**：CPU 92、streaming/cross CUDA 6、既有 CUDA batch/renderer 2。旧 FP 真实 fixture 占位项未运行，重复的早期检查未重复计数。

| 检查 | 实际结果 |
|---|---:|
| CPU context cached/full reference，长度 1/2/8/9/32 | 最大差 2.38e-7 |
| CPU 梯度 cached/full reference | 最大差 9.31e-10 |
| CUDA FP32 source+context 组合 reference | 最大差 3.34e-6 |
| CUDA BF16，所测组合用例 | 最大差 0 |
| 10,000 次 GPU temporal+cross 更新 | allocated 始终 103,622,656 bytes |
| B=1、W=8、BF16 纯 KV | 565,248 bytes = 0.5390625 MiB |

纯 context KV 增量为每流 8,192 bytes；上表不包含权重、元数据、workspace 或训练图。10,000 次检查使用 synthetic source token，未声称完整编码了 10,000 张真实图像。

测试覆盖公式与直接 MultiheadAttention 对照、未来帧隔离、流重置、padding、窗口淘汰、历史 context 梯度、原 source KV 不受新残差影响、无 GT 泄漏、无效输入/非有限 KV 不提交，以及零初始化后 Q/K/V 和 gate 能够学习。

真实 16 个连续训练片段、300 步过拟合：

| 指标 | 训练前 | 训练后 |
|---|---:|---:|
| Loss | 0.04983 | 0.02622 |
| 中心均值 | 3.24 mm | 2.12 mm |
| 旋转均值 | 2.48° | 1.33° |

完整 `8 burn-in +16 supervised` 的单卡显存探测通过；八卡 **50 步＋恢复 3 步**通过，恢复了全部 rank 的 optimizer、scheduler、RNG 和采样位置。新架构自己的预检批准记录已经生成，没有沿用旧 dual 的批准。

## 正式训练检查

保留加速后的配置：每卡 batch=8、累积1次、两个 loader worker、prefetch=2、pinned memory 和批量预处理。每个 optimizer step 为 **64 段、1,024 个监督帧**，W=8，8+16 unroll。BF16、原学习率分组、500 步 warm-up 和 1,000 步 RGB 冻结调度保持原设置。

正式训练前 100 步排除前四步后，实测 **1.0909 秒/步、938.64 个监督帧/秒**，峰值 allocated **10.46 GB/卡**。这是完整训练步耗时，不是单流推理延迟。

100 步 checkpoint 中已验证 attention Q、K、V、output projection 和 gate 参数都相对初值发生更新；最近 50 步 gate 均值约 **0.1200**，`gH` 范数均值约 **0.01340**，context KV 保留训练图。gate 不是置信度，也不代表已验证的抓取语义。

已另存首个检查到的正式 checkpoint：`runs/stream_v2_cross_8800_preflight/first_verified_formal_checkpoint.pt`，step=100，sampler position=6,400，SHA256 `90d687c4e9ea716b3c7f940681b698ac16213697817782affd701ecd9c6a706e`。已核实 optimizer、scheduler、八卡 RNG、架构与 cache contract。初期训练误差和预检成功不代表完整验证集精度优于旧 dual；已有 GT-depth 不一致未修复。

## 命令与证据

当前正式训练已启动，勿重复执行新训练命令。需要恢复时：

```bash
cd /mnt/why/dexycb_lip/stream_v2_cross_dual_8800
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_cross_8800.yaml \
  --resume runs/stream_v2_cross_from_dual_8800/last.pt \
  --output runs/stream_v2_cross_from_dual_8800 --max-steps 10000
```

完整评估入口支持该架构（以下未在本次执行）：

```bash
export PYTHONPATH="$PWD/src"
.venv-fp/bin/python -m lip.evaluate_stream \
  --config configs/stream_lip_v2_cross_8800.yaml \
  --checkpoint runs/stream_v2_cross_from_dual_8800/last.pt \
  --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 \
  --out runs/stream_v2_cross_s0_val
```

实现入口为 `src/lip/models/stream_cross_readout.py`；tracker、state/runtime、config/checkpoint、train/evaluate 和 preflight 已接入新架构。主要验证文件为 `tests/test_stream_cross.py`、`test_stream_cross_cuda.py`、扩展的 `test_stream_causality.py`。

- [机器可读完整汇总](../runs/stream_v2_cross_8800_preflight/summary.json)
- [预检批准](../runs/stream_v2_cross_8800_preflight/dual_cross/approval.json)
- [CPU 测试](../runs/stream_v2_cross_8800_preflight/dual_cross/cpu_tests.log)、[CUDA 测试](../runs/stream_v2_cross_8800_preflight/dual_cross/cuda_tests.log)
- [迁移逐参数记录](../runs/stream_v2_cross_from_dual_8800_init/weight_migration.json)
- [正式启动记录](../runs/stream_v2_cross_8800_preflight/launch.json)、[正式 rank0 快照](../runs/stream_v2_cross_8800_preflight/launch_snapshot/rank0.jsonl)
