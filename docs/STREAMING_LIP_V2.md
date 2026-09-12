# Streaming LIP v2：实现与实测交付

实现和短训练预检已通过。`stream_single` 与 `stream_dual` 均可独立迁移、训练、恢复和评估。当前短训权重的完整验证精度明显低于 V1，不能据此替换原模型；没有启动 10,000 步正式训练、上传 checkpoint 或推送 Git。

任务书来源为 `/home/haoyi/Documents/CODEX_TASK_STREAMING_LIP_V2.md`。本地仓库仍在 `fa0905147b4d7f581e25f6b075b600b3ea945bfe`，原有 FP-aware / basin / 加速实验的未提交改动全部保留。V2 新增文件尚未提交。旧 `Tracker`、旧训练和旧评估入口保留；唯一涉及公共旧工具的本次修改是 `evaluation/metrics.py` 增加可选 `dtype`，默认 float64 的旧行为不变，V2 显式请求 FP32 误差运算。

服务器原工程是运行副本，不是 Git checkout。本次使用独立候选目录测试，最终运行副本为 `/mnt/why/dexycb_lip/stream_v2`。日志和短训权重保留在 `/mnt/why/dexycb_lip/runs/stream_v2_preflight`，初始迁移权重保留在 `/mnt/why/dexycb_lip/runs/stream_v2_init`。运行副本通过链接复用既有环境和只读数据，没有覆盖服务器原 `src`。

**实现边界与代码。**

| 职责 | 文件 |
|---|---|
| 单帧编码、模型 API | `src/lip/models/stream_tracker.py` |
| 分层 source-only KV、完整前缀参考计算 | `src/lip/models/stream_attention.py` |
| single / object-context dual readout | `src/lip/models/stream_readout.py` |
| 当前 crop、9 通道几何、24 维状态和源图位置 | `src/lip/engine/stream_features.py` |
| 独立状态、functional cache、有界引用 ring | `src/lip/engine/stream_state.py` |
| initialize / step / commit / correct 生命周期 | `src/lip/engine/stream_runtime.py` |
| 完整片段 unroll 与缓存梯度 | `src/lip/engine/stream_training.py` |
| 配置、迁移与严格 resume | `src/lip/engine/stream_config.py`、`stream_checkpoint.py` |
| 连续片段采样 | `src/lip/data/stream_clips.py` |
| 训练与完整序列评估 | `src/lip/train_stream.py`、`evaluate_stream.py` |
| 迁移、预检、性能及统一验证 | `tools/migrate_lip_to_stream.py`、`preflight_stream.py`、`benchmark_stream.py`、`evaluate_stream_suite.py` |
| 后端核验、交付核验 | `tools/profile_stream_backend.py`、`finalize_stream_preflight.py` |

模型只编码当前 RGB-D crop，保留 ResNet50 layer3、原 geometry CNN、两层 spatial fusion、四层 pre-norm temporal Transformer 和六维解耦 pose head。每帧 source 为 16 个视觉 token 加 1 个状态 token。readout 只生成 query，不进入任何层的 K/V。当前 source 允许同帧双向注意力；显式帧块 mask 同时检查 frame ID、stream tag 和 key validity，SDPA 使用 `is_causal=False`。

相对时间差先用 float64 秒计算，再以 FP32 进入共享的时间 bias MLP。源 crop 的 A、K、base pose、坐标来源、时间和 silhouette fraction 在到达时固定。在线状态不保留历史 RGB-D 图像，不因新 base 重算旧 source。已有 DexYCB 索引只有 frame index 和官方 FPS，本次真实数据使用并明确记录这一时间来源；reader 支持缓存中的 `timestamps`，在线 API 和测试支持真正非均匀时间戳。没有把缺失的硬件时间戳伪称为已获取。

默认 functional cache 有界；另实现了经过等价测试的固定槽位引用 ring。它不是预分配的连续 CUDA K/V buffer，未声称已经完成这一可选优化。四层、W=8、B=1、BF16 的纯 K/V 实测为 **557,056 bytes = 0.53125 MiB**，single 和 dual 相同。此数不包含参数、元数据、workspace 或训练图。多层旧表示可携带更早上下文，不等于只依赖原始最近八帧。

dual 的 object readout 使用当时渲染 silhouette 的软 log bias，context readout 读取全部 crop source。state key 不加 object bias，空 silhouette 时该 bias 退化为零。context projection 零初始化，gate 初始 sigmoid(-2)，只是融合系数。没有手监督、orthogonality loss、FP refinement、basin loss、GT-depth 修正或校准置信度。

**权重迁移。**

用户明确选择归档 34,700 步：

```text
/mnt/why/dexycb_lip/runs/checkpoint_archive/checkpoint_34700.pt
SHA256 b3a7c4fd04d6d8c0fe24301dcfe97b460b508ce74a3d39651579badade2dc646
```

| 迁移 | 已迁移参数量 / 目标参数量 | 覆盖率 |
|---|---:|---:|
| V1 34,700 → stream_single | 15,574,854 / 15,648,142 | 99.53% |
| 短训 single 300 → stream_dual | 15,648,142 / 15,845,519 | 98.75% |

single 的 state MLP 虽然仍接收 24 维，也重新初始化；源图位置投影和时间 bias 是新模块。dual 新建 context query、context projection 和 gate，object query 从 single 复制。两个 query 同属一个参数张量，整块按新模块 LR 分组，迁移报告分别计数其中复制和新初始化的 256 个元素。逐参数 loaded/remapped/initialized/skipped 的完整原因见两个 `weight_migration.json`，没有用 `strict=False` 隐藏迁移缺失。

`--init-from` 新建 optimizer/scheduler、stage step 从 0 开始；`--resume` 检查 architecture、cache contract、配置、data hash、参数组及 world size，恢复 Adam、scheduler、各 rank RNG 和全局采样位置。在线 cache 不进入 checkpoint，加载权重或参数版本变化后旧 StreamState 被拒绝。

**真实运行的验证。**

最终回归为 **76 passed，1 deselected**。排除项是旧 `test_real_fp_requires_dependencies`：其源码本身是要求显式真实 fixture 的 skip 占位测试，V2 默认不运行 FP。旧模型、数据、loss、checkpoint、FP adapter mock 等回归包含在通过项中。

| 数值检查 | 实测最大绝对差 |
|---|---:|
| CPU FP32 cached vs full reference，长度 1/2/8/9/32，single/dual | 9.54e−7 |
| 训练梯度 cached vs full reference | 1.19e−7 |
| CUDA FP32 math，single | 1.55e−6 |
| CUDA FP32 math，dual | 1.43e−6 |
| CUDA BF16 math，所测 single/dual 用例 | 0 |

测试还覆盖不同 lane 长度、reset、全无效 padding、未来 RGB-D/label/time 改动、readout 不污染 source K/V、非零 mesh center、crop 投影、空/非有限深度、失效 proposal 不提交、外部修正、模型热更新，以及真正的 RNG/optimizer 续训。额外切断最后一帧 encoder 路径后，最终 loss 仍能经旧 K/V 回传至前面帧的 RGB encoder。连续 **10,000 次 GPU temporal 更新**，allocated 在第 100、1,000、5,000、10,000 步均为 **51,523,072 bytes**，纯 KV 均为 557,056 bytes。

| 16 个真实连续训练片段，300 steps | single 前 → 后 | dual 前 → 后 |
|---|---:|---:|
| loss | 0.18768 → 0.06517 | 0.09749 → 0.04385 |
| 中心误差均值 | 14.06 → 4.70 mm | 7.06 → 3.29 mm |
| 旋转误差均值 | 8.19° → 3.20° | 4.66° → 2.17° |

两者均另外跑过完整 `8 burn-in +16 supervised`、microbatch=2、accum=4 的单卡三步显存探测，以及 **8 卡 50 steps + checkpoint resume 3 steps**。每个 DDP step 为 64 段、1,024 个真实监督目标帧；sampler_position 在第 50 步为 3,200，恢复后第 53 步为 3,392。八个 rank 的恢复记录全部通过。single 单卡探测峰值 allocated 为约 3.08 GB，DDP 约 3.22 GB；dual 详细值和各 rank 日志见机器可读汇总。不能直接用新旧同一步数描述相同训练量。

训练 cache 保留 unroll 内梯度，只在 burn-in 边界 detach；pose feedback 每步 detach。每个片段后丢弃 cache，梯度累积完成后才更新参数。RGB 冻结阶段仍计算梯度以保持固定 DDP 参数集合，但 optimizer 前清除 RGB grads，因此不更新权重或 Adam 动量。BN statistics 始终冻结。

**性能实测与限制。**

H20、batch=1、一个固定验证对象；48 个新观测重复三次，每次排除前 8 次 warm-up，得到 120 个计时更新。core 使用预解码、预上传输入；端到端包括解码、H2D、正常状态检查和最终 pose D2H。没有 GT metrics、overlay 或磁盘写入。各 GPU 分项使用 CUDA events，每帧边界同步；后端 profiler 是独立的一帧检查，没有混入延迟数据。

| 路径 | 旧 V1 core 均值 | single core 均值 | dual core 均值 | single / dual 端到端 P95 |
|---|---:|---:|---:|---:|
| 完整 eager 对照 | 13.63 ms | 17.45 ms | 17.94 ms | 29.88 / 29.11 ms |
| 单独编译编码器实验，V1 保持 eager | 13.03 ms | 9.64 ms | 10.06 ms | 20.84 / 21.27 ms |

**eager 没有加速。** 图片数从每更新 8 张降到 1 张，render 都是每更新 1 次，但小算子和事务检查开销抵消了计算量收益。完整前缀 reference 只测 16 个新帧作为诊断，没有用它较慢的耗时夸大生产加速。

编译仅覆盖 `encode_current`，temporal 和状态 API 保持 eager。第一次编译/预热约 135 s，第二模型复用缓存约 0.30 s，均在计时前完成。编译后的 BF16 encoder 与 eager 最大差为 single 0.01935、dual 0.02123；没有放宽 FP32 KV 测试阈值来掩盖它，也没有完成编译版本全量精度对照，因此它是 `benchmark_stream.py --compile-stream` 的独立实验开关，默认训练/模型仍是 eager。

实际 profiler 确认 spatial fusion 调用了两次 `aten::_scaled_dot_product_flash_attention`，temporal 调用了四次 `aten::_scaled_dot_product_efficient_attention`。记录了 mean/P50/P95/P99、allocated/reserved、各分项、实际图片数、KV bytes 和基于实测 service time 的不丢帧队列 age。该队列是模拟，并非真实摄像头在线实验；不能把这些数字作为所有对象/距离都达到 30/60 Hz 的保证。

**完整 s0 val 的初步精度。**

三个分支各自完整评估了 **320 流、23,200 帧**，核对相同 `(stream_id, frame_id)` 集合；只用首帧 GT 初始化，后续不做 GT 重置、不删失跟帧。下表排除 320 个初始化帧，成功率与均值采用物体宏平均。

| 权重 | ADD@0.1d | ADD-S@0.1d | 中心均值 | 旋转均值 | lost |
|---|---:|---:|---:|---:|---:|
| V1 34,700 standalone | 81.36% | 96.92% | 10.02 mm | 11.64° | 2.25% |
| single 短过拟合 300 | 44.16% | 67.89% | 65.63 mm | 30.79° | 22.39% |
| dual 从 single 300 再短过拟合 300 | 45.04% | 69.41% | 78.00 mm | 32.06° | 21.76% |

0.05d 阈值、中心/旋转 median/P95、每相机/物体、`moving AND visibility<0.3` 交叉子集和恢复统计都在各自 metrics.json。dual 有额外 warm-start 训练量，不能把这张表解释为匹配预算的架构因果消融。它也没有全面优于 single，因此没有把 dual 设为默认。两份 v2 权重都还不具备替换 V1 的精度；完整数据集的正式训练与最终精度验证仍未完成。W=1 无历史对照需要单独训练，本次没有制造未训练的对照成绩。官方 test 未访问。

现有 `visibility` 衡量画面内遮挡比例，不包含出画截断；这一已有局限没有被隐藏。旧的 camera-dependent GT-depth 不一致仍存在，本次另抽查了 16 个 train 片段的 GT render 残差，并完整记录，没有修改 GT 或深度来改善数字。见 [此前几何排查](rgbd_alignment_audit.md)。

**在线接口。** RGB 为 CHW、depth 为 1HW 浮点米，时间戳使用秒，pose 是 object-to-camera；外部初值使用原始 mesh 原点。

```python
model = make_model(config).cuda().eval()
load_init(checkpoint, model, data_audit, config)
renderer = Renderer('cuda')
state = model.initialize(T0_original, mesh, K, stream_id, timestamp0,
                         object_id=object_id, camera_id=camera_id,
                         mesh_hash=mesh_hash, image_shape=(H, W))
proposal, pending = model.step(rgb_t, depth_m_t, timestamp_t, state,
                               renderer=renderer, precision='bf16')
if proposal['status'] == 'ok':
    state = model.commit(proposal, pending)
# needs_reinit 要求调用者提供可靠新初值，再 initialize / correct(..., relocalization=True)。
# 小外部修正调用 correct(state, corrected_original_pose)，不重写旧 source。
```

时间倒退、默认 >0.5 s 间隔、流/物体/camera/mesh/分辨率/K 变化或权重版本变化都会明确要求重初始化。非有限 proposal 不提交 pose 或 KV。空深度、几何不可靠等单独记录，不能把这些检查当作已校准失跟概率。crop 脱离目标时仍可能失跟，没有自动全局重定位。

**运行命令。** 服务器使用已验证的独立运行目录和既有 Python；下面仅是交付命令，本次没有执行正式长训。

```bash
cd /mnt/why/dexycb_lip/stream_v2
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
export LIP_INIT_CKPT=/mnt/why/dexycb_lip/runs/checkpoint_archive/checkpoint_34700.pt
export LIP_STREAM_PYTHON=/mnt/why/dexycb_lip/.venv-fp/bin/python
export PYTHONPATH="$PWD/src"

# 使用已保留的迁移权重，开始新的 single 正式 stage
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_single.yaml \
  --init-from runs/stream_v2_init/init.pt \
  --output runs/stream_v2_single --max-steps 10000

# 同一架构/配置真正续训
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_single.yaml \
  --resume runs/stream_v2_single/last.pt \
  --output runs/stream_v2_single --max-steps 10000

# 已预检的独立 dual 初始权重来自 single 的短训分支
bash scripts/train_stream_8gpu.sh \
  --config configs/stream_lip_v2_dual.yaml \
  --init-from runs/stream_v2_init/dual/init.pt \
  --output runs/stream_v2_dual --max-steps 10000

# 单卡完整验证；只用首帧 GT
"$LIP_STREAM_PYTHON" -m lip.evaluate_stream \
  --config configs/stream_lip_v2_single.yaml \
  --checkpoint runs/stream_v2_single/last.pt \
  --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 \
  --out runs/stream_v2_single_val

# 八卡统一评估：三者均为 standalone，仅首帧 GT，输出目录必须尚不存在
"$LIP_STREAM_PYTHON" tools/evaluate_stream_suite.py \
  --legacy-checkpoint "$LIP_INIT_CKPT" \
  --legacy-config configs/resolved_8gpu.yaml \
  --single-checkpoint runs/stream_v2_single/last.pt \
  --dual-checkpoint runs/stream_v2_dual/last.pt \
  --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 \
  --out runs/stream_v2_full_training_comparison
```

从将来正式训练好的 single 重新迁移 dual，应使用新的输出路径和新的预检目录：

```bash
"$LIP_STREAM_PYTHON" tools/migrate_lip_to_stream.py \
  --checkpoint runs/stream_v2_single/last.pt \
  --config configs/stream_lip_v2_dual.yaml \
  --out runs/stream_v2_dual_from_full_single_init/init.pt

"$LIP_STREAM_PYTHON" tools/preflight_stream.py \
  --config configs/stream_lip_v2_dual.yaml \
  --init-from runs/stream_v2_dual_from_full_single_init/init.pt \
  --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 \
  --out runs/stream_v2_dual_from_full_single_preflight
```

正式训练的配置 `preflight_receipt` 应指向相应新目录的 `dual/approval.json`；该路径不参与训练配置 hash。更改数据、架构、缓存契约、学习配置或代码后，旧 preflight 不能自动沿用。`--init-from` 与 `--resume` 互斥；非正 `--max-steps` 明确拒绝，避免 0 被错误解释成默认长训练。

统一八卡分片比较由 `tools/evaluate_stream_suite.py` 提供；各分片通过后合并并核对完整帧集合。`benchmark_stream.py` 提供 eager/reference/编译性能开关。`preflight_stream.py` 自动完成 CPU 检查、真实片段短训、完整 unroll 显存、GPU 等价与内存稳定性、8 卡短跑及恢复；发现其他 GPU 计算进程时记录未运行项，不抢占或杀外部任务。

证据入口：

- [机器可读交付汇总](../runs/stream_v2_preflight/summary.json)、[精度/延迟/短训图](../runs/stream_v2_preflight/summary.png)
- [基线审计](../runs/stream_v2_preflight/baseline_audit.json)、[远端原始环境](../runs/stream_v2_preflight/remote_environment.json)
- [最终测试日志](../runs/stream_v2_preflight/delivery_tests.log)、[JUnit](../runs/stream_v2_preflight/delivery_tests.xml)
- [single 预检](../runs/stream_v2_preflight/single/approval.json)、[dual 预检](../runs/stream_v2_preflight/dual/approval.json)
- [eager 性能](../runs/stream_v2_preflight/benchmark_eager_all/benchmark.json)、[编译实验](../runs/stream_v2_preflight/benchmark_compiled/benchmark.json)、[实际 attention 后端](../runs/stream_v2_preflight/attention_backend.json)
- [三分支完整验证](../runs/stream_v2_preflight/validation/comparison.json)
- [single 迁移记录](../runs/stream_v2_init/weight_migration.json)、[dual 迁移记录](../runs/stream_v2_init/dual/weight_migration.json)
- [独立运行目录验证](../runs/stream_v2_preflight/runtime_verification.json)、[交付哈希核对](../runs/stream_v2_preflight/delivery_receipt.json)

初次旧回归 collection 曾因隔离候选目录尚未复制两个旧工具而失败，补齐后重跑通过；失败日志保留。DDP 有 PyTorch/NCCL 的 device_id 提示，已显式设置各 rank 当前 CUDA device，实测没有挂起。最终源码 hash 为 `3d9cd99903161b0ee7b17cc50df80afe5204cb62c0a3674961f87fd161d9d47e`。交付核验通过逆向恢复审计补丁，重建并匹配 single/dual 预检的原始源码 hash，再结合最终回归更新批准记录，没有静默替换旧的审计来源。
