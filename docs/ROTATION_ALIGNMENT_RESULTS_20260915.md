# 专用旋转对齐分支：完成结果

两组各 1,000 步训练、完整 native s0 val、独立延迟测量和归档均已
完成。**新增 14×14 CAD 几何旋转对齐分支没有证明相对同预算对照
的增益，未晋级；保留旧 residual 权重。** 当前八卡空闲，没有新的
official test 或额外训练进程。

每个模型均使用相同的真实 PoseCNN 初值，320 流 / 23,200 帧，
每帧一次 LIP 校正，全程零 FP。下面对照是本轮相同真实初值训练
配方的 control，不是旧父模型，也不是上一轮的 control。

| 指标 | 同预算 control | 新 alignment | 旧父模型 | FP |
|---|---:|---:|---:|---:|
| 总体 ADD <0.1d (%) | 57.744 | 57.843 | 57.194 | 52.189 |
| 总体 ADD-S <0.05d (%) | 84.481 | 84.480 | 83.664 | 72.419 |
| 严重遮挡 ADD-S <0.05d (%) | 32.182 | 32.195 | 29.278 | 21.389 |
| 坏初值前八帧 ADD <0.1d (%) | 56.302 | 56.302 | 54.967 | 63.845 |
| 坏初值前八帧 ADD-S <0.05d (%) | 80.512 | 80.400 | 79.494 | 82.600 |
| 坏初值前八帧旋转误差 (°) | 44.387 | 44.408 | 44.102 | 41.151 |
| 坏初值前八帧中心误差 (mm) | 13.091 | 13.069 | 13.984 | 14.479 |

新增分支减 control 的配对差值（40 条物理序列、2,000 次共享
bootstrap，未做多重比较校正）：

- 总体严格 ADD-S：−0.000479 pp，95% CI [−0.129, 0.131]。
- 严重遮挡严格 ADD-S：+0.013391 pp，95% CI [−0.244, 0.298]。
- 坏初值前八帧旋转误差：+0.020382°，95% CI [−0.013, 0.045]。
- 坏初值前八帧中心误差：−0.022296 mm，95% CI [−0.098, 0.045]。

新分支相对旧父模型总体严格值 +0.816 pp，但同配方 control 也有
相近收益，不能把这部分收益归因于新结构。预设四项精度/中心保护
均通过，启动旋转相对父模型与 control 的两个配对条件均未通过。
没有事后放宽晋级规则。

分支并未完全关闭：22,303 次跟踪更新均记录到非零旋转残差，逐帧
算术平均 0.118493°，中位 0.106535°，最大 1.503360°。这是分支
自身的额外旋转，不是最终旋转误差或总校正量；不能仅据幅度小就
断言梯度死区或解释全部失败。此前真实 pilot 已证实 CAD 编码器
收到非零梯度。下一步应先检查启动时分支贡献、匹配特征及梯度，
再决定训练目标或结构修改；没有自动开启新的参数扫描。

H20 batch1、固定 20 类流、每流 32 次更新、CPU RGB-D 到 CPU pose、
排除 IO 的独立测量；steady 为排除首八次后的 480 次更新：

| steady 耗时 | control | alignment |
|---|---:|---:|
| 平均 | 22.236 ms | 24.921 ms |
| P50 | 22.101 ms | 24.689 ms |
| P95 | 23.483 ms | 26.714 ms |

新分支 P50 增加 2.588 ms（约 11.7%），目前没有对应的可证实精度
收益。上述数值不是含 IO 的端到端流平均，也不是官方 BOP AR。

核验：52 CPU、4 CUDA、7 统计/边界和 26 FP 协议测试通过；真实
父模型零起点、梯度、显存、DDP 保存恢复检查通过。当前源码下旧
父模型的完整 pose 和阈值分数逐位复现。预测 JSON 新增诊断字段，
因此文件 SHA 与旧版本不同，不能称整个 JSON 文件逐字节相同。

所有控制器终态完成。完整归档已回传，本次重新核验 archive SHA
和全部 1,296 个文件：
`a60c6689d799f9666221a86677f4bf7b58f049e0ea65e91c18305791b6db8142`。

- 保留父权重：`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`。
- 新 alignment final1000：`014e13710125daf7b103f5b1050beb9e8876eb86036ff4dfdff5973019af9bb7`。
- 服务器结果：`/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs/alignment_val/`。
- 训练日志：同根目录 `../alignment_pair/{control,alignment}/train/`。

[完整表格与图像](../runs/intraframe_alignment_20260915_completion/package/evidence/experiment/runs/alignment_completed/figures/report.md)。
[所有配对区间及晋级检查](../runs/intraframe_alignment_20260915_completion/package/evidence/experiment/runs/alignment_val/comparison/comparison.json)。
[最终测试/实验核验](../runs/intraframe_alignment_20260915_completion/package/evidence/experiment/runs/alignment_completed/verification.json)。
[本地文件核验](../runs/intraframe_alignment_20260915_completion/package/verification.json)。

第一阶段的冻结双次校正结论仍成立：启动成功率/中心有改善，但
启动延迟近翻倍、严重遮挡严格点值略降，保持一次校正默认。
本轮两个阶段均未建立“所有指标超过 FP”的结论。
