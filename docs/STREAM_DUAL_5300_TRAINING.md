# 从 single 5,300 步迁移 dual 并加速训练

**正式 dual 已于北京时间 2026-09-12 10:13:38 启动。** 10:15:46 的交付快照已核实第 **100 步** checkpoint、完整 optimizer/scheduler/八卡 RNG 和采样位置 6,400，训练继续运行。前 100 步排除前四步后的实测均值为 **1.0277 秒/步、996.37 个监督帧/秒**，相对原配置基线约 **3.57 倍**；100 步时按这一速度估计还需约 2.83 小时，实际时间会随数据读取与后续阶段变化。

首个正式 checkpoint 另存于 `runs/stream_v2_dual_5300_tuning/first_formal_checkpoint.pt`，SHA256 为 `470512ea88555db977a0a6ecfeae88c28948ff1d44843fa2631cd1124c996c56`。该快照证明已启动且能够保存恢复状态，不代表正式训练已完成或验证精度已经提升。

用户要求在下一个 checkpoint 保存后转训 dual，并提高硬件利用率。本次在 single 的 **5,300 步**原子 checkpoint 完成后保留该文件的独立硬链接，核实 scheduler、optimizer 和八个 rank 的 RNG，再向指定 torchrun launcher 发送 SIGTERM。没有停止其他任务。

- single 归档：`/mnt/why/dexycb_lip/stream_v2/runs/stream_v2_single_archive/checkpoint_5300.pt`
- SHA256：`1877627fd31bf0a1c86a3b6ca3546608818639694642d896a7db1e348383b96b`
- sampler position：339,200；single 可从该位置恢复。
- dual 初值：`runs/stream_v2_dual_from_single_5300_init/init.pt`
- dual 初值 SHA256：`42e5628f4d3cbfbd4c72d000a625a8adce516fc70b5b949022306dd53c34ceb8`
- 迁移参数：15,648,142 / 15,845,519（98.7544%）。新建 context query、context projection 和 gate；这是新 stage 的 warm-start，不恢复 single optimizer。

旧 single 运行目录 `/mnt/why/dexycb_lip/stream_v2` 的源码 hash 仍为 `3d9cd99903161b0ee7b17cc50df80afe5204cb62c0a3674961f87fd161d9d47e`。新代码单独部署到 `/mnt/why/dexycb_lip/stream_v2_dual_5300`，源码 hash 为 `5ccca80a97c2a50a00819dc3d8d6f86c953eae1e8c7a799e45f85330340708e2`。新目录通过链接复用既有环境、数据和 runs；旧批准记录未改写，旧模型和已有 FP/basin 实验保留。工作区未提交、未推送 Git，没有上传 checkpoint，也没有改驱动或全局 CUDA。

## 实际修改

- `configs/stream_lip_v2_dual_5300_fast.yaml`：每卡 batch=8，累积一次，两个 loader worker，prefetch=2，pinned memory。
- `src/lip/engine/stream_batch_features.py`：批量 crop、geometry channel、位置及 24 维状态构造；复用原有 crop bounds、mesh renderer 和 pose math。
- `src/lip/engine/stream_training.py`：可选批量预处理。每段原始 RGB/depth 上传一次，保留 uint8/uint16，逐帧消费；未来观测不进入当前编码、pose 或 KV。
- `src/lip/train_stream.py`：可配置 pin_memory，旧配置默认行为保持不变。
- `tests/test_stream_batch_features.py`、`test_stream_causality.py`：特征、预测、梯度、未来观测和 GT 隔离、旧 KV 梯度回传测试。
- `tools/pause_stream_at_checkpoint.py`、`tune_stream_training.py`、`verify_stream_batch_real.py`：保存后暂停、完整八卡计时、真实片段数值对照。

有效 batch 始终为 **64 段 / optimizer step**，每步 **1,024 个监督帧**。保留 8 burn-in +16 supervised、W=8、BF16、原学习率、warm-up 和冻结调度；未减少 rollout、关闭梯度、加入 FP/critic 或修改 GT-depth。训练步数为 dual 自己的 stage step，不能与 single 5,300 步直接相加当作同一 optimizer 轨迹。

## 同条件吞吐

同一 dual 初始化、同一训练采样、8 张 H20，每组 16 步，排除前四步，以每步最慢 rank 的记录计时。所有数字包括取数据、完整 burn-in/unroll、backward、梯度同步与 optimizer；不把初始化和最后退出写 checkpoint 的时间计入训练步耗时。

| 配置 | 秒/步 | 监督帧/秒 |
|---|---:|---:|
| 原路径，每卡 2 × 累积4，1 worker | 3.6713 | 278.92 |
| 原路径，每卡 4 × 累积2，2 workers | 2.3032 | 444.61 |
| 原路径，每卡 8 × 累积1，2 workers | 1.6815 | 608.97 |
| 批量预处理，每卡 8 × 累积1，2 workers | **0.9766** | **1,048.55** |
| 批量预处理，每卡 8 × 累积1，4 workers | 0.9852 | 1,039.41 |

选择两个 worker，相对原配置 **3.759 倍**吞吐。实测峰值 allocated 为 10,452,965,888 bytes/卡（约 10.45 GB）。纯训练采样区间的 GPU utilization 均值约 43.7%，原配置约 30.5%；未宣称 GPU 已满载。逐 mesh rasterization 和因果时序依赖仍有开销，没有通过扩大有效 batch 或占满显存来制造优化成绩。正式运行会另报稳定步耗时。

## 数值检查与已知差异

CPU/CUDA 混合 lane 的特征对照最大差为 0；CPU 完整训练梯度对照最大差为 0。16 个真实训练片段、每批 8 段、完整 8+16 unroll 的 FP32 math 对照中，预测、loss、梯度最大差均为 **0**。

生产 BF16 路径的真实预测和 loss 也逐值相同，但 backward 不是逐位确定的。第一组直接比较的梯度相对 L2 约 0.001063，略超原定 0.001 阈值，因此检查最初失败，日志保留。随后增加原路径重复运行对照：两个 batch 的原路径自身波动分别为 0.001066 / 0.000974；新批量路径分别为 0.001089 / 0.000991，处于相同数量级。另用 FP32 math 严格复核为零，没有修改模型或 FP32 容差来掩盖该差异。

新增路径的未来 RGB-D、未来时间和 GT 不泄漏测试，以及仅经历史 KV 回传到更早 RGB encoder 的测试均通过。现有 camera-dependent GT-depth 不一致未修复。短训误差下降不是完整验证集精度提升的证据，本次没有新的 dual 全量验证成绩。

最终源码共覆盖 **81 个不同测试并通过**：完整 CPU 集合 76、原 streaming CUDA 集合 3、新 CUDA batch 特征 1、旧 CPU/CUDA renderer 一致性 1。FP 真实 fixture 占位测试未运行；早先 7 项 CPU 新路径检查与最终 CPU 集合有重叠，没有重复计数。

最终配置的真实 16 段、300 步短过拟合完成：loss **0.07114→0.02292**，中心误差 **4.522→1.628 mm**，旋转误差 **3.524→1.208°**。另外通过完整 unroll 单卡显存探测、10,000 次 CUDA temporal 更新的有界显存检查，以及 **八卡 50 步＋恢复 3 步**；八个 rank 的 optimizer、scheduler、RNG 和采样位置均恢复。新配置自己的 `approval.json` 已批准，未沿用旧 single 或旧 dual 配置的批准。

## 运行与恢复

dual 使用独立的新 stage，目标 10,000 步。下面命令与最终配置对应；不要在已有训练进程运行时重复启动。

```bash
cd /mnt/why/dexycb_lip/stream_v2_dual_5300
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910

# 新 dual stage
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_dual_5300_fast.yaml \
  --init-from runs/stream_v2_dual_from_single_5300_init/init.pt \
  --output runs/stream_v2_dual_from_single_5300 --max-steps 10000

# 恢复 dual
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_dual_5300_fast.yaml \
  --resume runs/stream_v2_dual_from_single_5300/last.pt \
  --output runs/stream_v2_dual_from_single_5300 --max-steps 10000

# 如需恢复 single，使用保留原源码及批准记录的旧运行目录
cd /mnt/why/dexycb_lip/stream_v2
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_single.yaml \
  --resume runs/stream_v2_single_archive/checkpoint_5300.pt \
  --output runs/stream_v2_single --max-steps 10000
```

证据保存在共享 runs 下的 `stream_v2_dual_5300_tuning/`、`stream_v2_dual_5300_preflight/` 和 `stream_v2_dual_from_single_5300/`；最后一个目录为正式训练。

- [正式启动与全部实测汇总](../runs/stream_v2_dual_5300_tuning/summary.json)
- [新配置预检批准](../runs/stream_v2_dual_5300_preflight/dual/approval.json)
- [真实 FP32 数值检查](../runs/stream_v2_dual_5300_tuning/batch_real_fp32.json)
- [BF16 重复对照](../runs/stream_v2_dual_5300_tuning/batch_real_numeric_repeat.json)
- [CPU 测试日志](../runs/stream_v2_dual_5300_preflight/dual/cpu_tests.log)
