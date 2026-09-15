# 真实初值训练与同条件 FP 对比：完成结果

两组固定 1,000 步训练、三份 LIP 权重的完整 native s0 val、独立速度
测量和归档均已完成。**真实初值混合训练改善了总体严格精度和启动
中心恢复，但没有解决启动旋转短板；按预先固定的规则未晋级，仍保留
旧 residual。** 真实初值组权重单独保留，不能把“未晋级”解释为没有收益。

## 完整验证

每份权重均评估 320 条流、23,200 帧。所有方法使用同一份冻结的真实
PoseCNN 初值；LIP 全程零 FP，纯 FP 使用 `track_one(iteration=2)`。
578 个无初值帧仍为失败。以下均为 object-macro native val 分数，
不是官方 test BOP AR；visibility 是 native scoring proxy。

| 指标 | FP | 旧 residual | noisy-GT 对照 | 真实初值混合 |
|---|---:|---:|---:|---:|
| 总体 ADD <0.1d (%) | 52.189 | 57.194 | 56.757 | **57.779** |
| 总体 ADD-S <0.05d (%) | 72.419 | 83.664 | 83.110 | **84.489** |
| visibility <0.5，ADD <0.1d (%) | **37.553** | 27.163 | 26.967 | 27.586 |
| visibility <0.3，ADD <0.1d (%) | **14.685** | 12.219 | 11.619 | 12.359 |
| visibility <0.3，ADD-S <0.05d (%) | 21.389 | 29.278 | 30.229 | **32.099** |
| 坏初值前八帧，ADD <0.1d (%) | **63.845** | 54.967 | 52.347 | 56.168 |
| 坏初值前八帧，ADD-S <0.05d (%) | **82.600** | 79.494 | 75.301 | 80.746 |
| 坏初值前八帧，旋转误差 (°，低好) | **41.151** | 44.102 | 44.610 | 44.408 |
| 坏初值前八帧，中心误差 (mm，低好) | 14.479 | 13.984 | 15.329 | **13.090** |

两组新训练的父模型、64,000 个片段、遮挡随机种子、初始观察处理、
fresh optimizer 和 1,000 步预算相同。真实组实际采用 30,956 次真实
初值，1,168 次请求因检测缺失显式回退。RGB 冻结、其他模块适配。
原有 residual 是历史父权重，不能把新训练与父权重的差异都归因于
“是否真实初始化”；真实组与 noisy-GT 组才是本轮预算匹配的对照。

## 收益与仍未解决的问题

40 条物理序列、2,000 次共享配对 bootstrap；区间未做多重比较校正，
也不代表多训练种子的稳定性。

| 真实组 − 参照 | 指标 | 差值 | 95% CI |
|---|---|---:|---:|
| 旧 residual | 总体严格 ADD-S | +0.825 pp | [+0.038, +1.649] |
| noisy-GT 对照 | 总体严格 ADD-S | +1.380 pp | [+0.660, +2.090] |
| 旧 residual | 严重遮挡严格 ADD-S | +2.820 pp | [−0.654, +5.176] |
| noisy-GT 对照 | 严重遮挡严格 ADD-S | +1.870 pp | [+0.331, +3.155] |
| 旧 residual | 坏初值前八帧严格 ADD-S | +1.252 pp | [−0.298, +2.484] |
| noisy-GT 对照 | 坏初值前八帧严格 ADD-S | +5.445 pp | [+3.369, +7.412] |
| 旧 residual | 坏初值前八帧中心误差 | −0.894 mm | [−1.686, −0.208] |
| 旧 residual | 坏初值前八帧旋转误差 | +0.306° | [−0.414, +0.936] |
| FP | 严重遮挡严格 ADD-S | +10.710 pp | [+1.785, +21.518] |
| FP | 坏初值前八帧 ADD | −7.677 pp | [−11.141, −3.317] |
| FP | 坏初值前八帧旋转误差 | +3.257° | [+1.785, +4.577] |

真实组的中心恢复有所改善，但旋转仍较 FP 慢。相对旧 residual，
visibility <0.5 的旋转误差点值还从 58.390° 上升到 60.917°；差值
区间 [−0.445°, +4.323°] 包含零，不能声称已证明旋转退化，也不能
把更好的 ADD-S 自动解释为旋转变好了。

预设晋级条件要求保护总体 ADD、总体严格 ADD-S 和严重遮挡严格
ADD-S，同时至少一个预列 FP 短板相对旧 residual 获得有利配对区间。
真实组通过前三项，最后一项未通过。noisy-GT 对照还未通过两项总体
保护。因此选择仍为 `residual`，没有事后放宽门槛。

下一轮应保留真实初值与启动遮挡配方，单独对启动旋转校正做预算
匹配的干预，并检查中心优势和全程旋转误差；本轮单种子结果不足以
确定更强的旋转 loss 或更大的网络一定有效。当前没有新训练进程，
没有启动新的 official test。

## 独立速度和显存

同一 H20、batch1、20 类固定流，每流 32 次更新。输入为 CPU RGB 和
米制 depth，输出为 CPU original-mesh pose，CUDA 同步；包含预处理、
render、数据传输和时序更新，排除 IO、模型/mesh 加载与首次 priming。
以下 steady 指排除每流最初八次更新后的 480 次更新。

| 指标 | FP | 保留的旧 residual |
|---|---:|---:|
| steady 平均耗时 (ms) | **21.624** | 21.997 |
| steady P50 (ms) | **20.322** | 22.041 |
| steady P95 (ms) | 31.591 | **23.291** |
| PyTorch 峰值 allocated (bytes) | 4,343,629,312 | **142,623,232** |
| 每流末设备显存采样最大值 (MiB) | 4,914 | **904** |

LIP 的 P95 和已记录显存更好，FP 的中位延迟更低。不能概括为 LIP
所有速度指标都快。PyTorch 峰值不含全部 native CUDA/rasterizer
分配；设备显存数值是每流结束采样最大值，不是连续捕获的真正峰值。
未晋级的真实组没有额外实测延迟，不能拿旧 residual 的数值冒充。

## 核验与完整证据

- 两组均完成 1,000 步、3,072,000 个监督目标；八份 rank 日志、
  optimizer/scheduler、采样位置、RNG 数量与有限值检查通过。
- 49 CPU + 4 CUDA + 7 统计/边界 + 26 FP 协议测试通过；真实接口、
  梯度 pilot、显存探测和 DDP 保存恢复已通过。
- 旧 residual 在当前源码下完整 23,200 帧预测逐字节复现，所有
  阈值分数一致，排除了本轮训练数据改动改变其推理行为的疑虑。
- 所有控制器已完成，当前八卡空闲。完整归档已下载，本次复核了
  archive SHA 与全部 **1,246** 个文件的 SHA。
- 归档 SHA：`db3db13eac18a001f4647909e046f17f5961f26580ca4e06e5d44b3beb649d92`。

权重 SHA（服务器二进制保留）：

- 当前选择：`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`。
- 真实初值 final1000：`23a5233f3c2e0df69cefab0276ed27a93954c090fb792d9e5727aedf72d253a8`。
- noisy-GT final1000：`eef19a679f90866b400f8b44df813c50b99f09e5a7fbf9b60320dbc51c0af92d`。

服务器结果根目录：
`/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/`。
两组训练在 `pair/{control,real_mix}/train/`；验证与冻结选择在
`full_val_v2/`；最终核验在 `delivery_completed/`。

- [完整图表与四组报告](../runs/fp_real_init_20260915_completed/evidence/experiment/runs/delivery_completed/figures/report.md)。
- [所有配对区间与预设选择检查](../runs/fp_real_init_20260915_completed/evidence/experiment/runs/full_val_v2/comparison/comparison.json)。
- [逐帧配对 CSV](../runs/fp_real_init_20260915_completed/evidence/experiment/runs/full_val_v2/comparison/paired_frames.csv)。
- [物理序列统计 CSV](../runs/fp_real_init_20260915_completed/evidence/experiment/runs/full_val_v2/comparison/paired_physical_sequences.csv)。
- [最终测试与实验凭据](../runs/fp_real_init_20260915_completed/evidence/experiment/runs/delivery_completed/verification.json)。
- [本地归档核验](../runs/fp_real_init_20260915_completed/verification.json)。

以上是 single-seed native val 的完成结果，不能据此声称所有指标
超过 FP，亦不构成 official benchmark 的 SOTA 结论。
