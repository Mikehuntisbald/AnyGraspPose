# RA-L / ICRA 新颖性判断（2026-09-11）

这是针对当前实现和近邻论文的判断，不是穷尽检索、首创认证或录用概率预测。

## 当前判断

方向有成为 RA-L / ICRA 论文的潜力，但当前还没有把关键新颖性与主模型收益证明完整。
ResNet + geometry CNN + temporal Transformer、无需 MANO、混合 rollout，以及增加两个 critic 输出头，本身不足以构成强方法贡献。
最值得做实的是：利用冻结 refiner 的真实执行结果，学习观测与候选位姿共同决定的有限步可恢复区域，再用该监督训练时序初始化器，改善后续精修与闭环恢复。

## 必须正面对比的先例

| 先例 | 已经覆盖的思想 | 当前方案需要明确的区别 |
|---|---|---|
| [se(3)-TrackNet, IROS 2020](https://arxiv.org/abs/2007.13866) | 当前 RGB-D、上一估计位姿下的 CAD 渲染、相对位姿更新和长期跟踪 | 多帧状态如何改善可恢复性，不能只称使用历史是创新 |
| [MegaPose, CoRL 2022](https://proceedings.mlr.press/v205/labbe23a.html) | 对候选位姿分类，判断它能否被 refiner 校正；论文明确采用 basin-of-attraction 表述 | 实际执行固定 refiner 得到 outcome，与其按训练扰动分布生成正负标签不同；还需证明把 critic 用作时序 actor 训练信号的独立收益 |
| [Pose Proposal Critic, 2020](https://arxiv.org/abs/2005.06262) | 学习观测与渲染之间的重投影误差，用于位姿 refinement | 预测当前几何误差，与预测后续固定算法执行结果不同；不能把 critic 这个名称当创新 |
| [FoundationPose, CVPR 2024](https://arxiv.org/abs/2312.08344) | 通用物体位姿估计与跟踪，使用 CAD 或参考图像，不需针对新物体微调 | 当前 LIP 使用 DexYCB 训练，不能把击败零样本 FP 当作已经隔离出 basin 的贡献 |

MegaPose 的训练细节见[原论文 3.2 节](https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf)：正例来自 refiner 训练时的扰动分布，远离该区域的候选作负例。因此当前真实 outcome 标注与其有具体区别，但“学习是否落入 refinement basin”不是空白问题。

通过 critic 对连续动作的导数更新 actor 也有成熟先例，例如 [Deterministic Policy Gradient, ICML 2014](https://proceedings.mlr.press/v32/silver14.html)。当前实现是监督训练的有限步 outcome critic，没有 Bellman 价值学习；不应因使用 actor/critic 名称就宣称新 RL 算法。

## 可以成立的核心主张

> 通过学习冻结位姿精修器在固定计算预算下的可恢复区域，训练时序先验生成更适合后续精修的初始化，提升遮挡后恢复和长期闭环稳定性。

“可恢复区域”应限定观测、refiner 版本、迭代次数和成功度量。当前跑两次 refinement，并不支持渐近收敛或全局稳定性保证。

关键区别应同时体现在：真实执行结果监督、critic 指导上游初始化器、与部署一致的 post-FP 状态递推。这个组合是待证实的贡献，不是已完成的首创主张。

## 最优先补齐的证据

1. **因果消融。** 相同数据、训练预算和 FP 设置，比较普通 LIP+FP、仅 FP-aware rollout、rollout+basin。另保留纯 FP、恒速先验+FP。只比较最终方法与冻结零样本 FP 不足以排除任务训练的作用。
2. **真实可恢复性。** 控制初始 ADD/旋转/中心误差范围，再比较精修后成功率；比较等 refine 次数与等延迟预算。避免把更接近 GT 的普通预测进步误称为学会了 basin。
3. **梯度有效性。** 在未用于 critic 拟合的样本上检查：沿 critic 建议的小步移动，真实 FP 后误差是否改善。AUROC、校准和候选排序不能单独证明梯度可用于优化 actor。
4. **闭环与错误模式。** 报告遮挡后恢复率、连续失败时长、旋转误锁、ADD 与 ADD-S，区分合法对称性与任务相关方向。当前果冻盒近 90 度误锁仍通过 ADD-S 的问题必须处理或明确限制。
5. **泛化与机器人价值。** 除 DexYCB 外，增加独立场景/运动分布、多个随机种子和按真实序列计算的不确定性；尽可能在第二个 refiner 上复验。若物体仍是同一组 YCB CAD，不能称为未见物体泛化。机器人实验应测量跟踪错误对抓取、放置或交接成功的影响。

当前已安排的 31k 同 checkpoint 全量 LIP 与 LIP+FP 比较可以验证组合系统的效果，但不能替代上述 basin 消融。即使组合变好，也尚不能说明增益来自 critic。

## 投稿定位

这是个人的论文成熟度判断，不是会议规定：若主要结果仅是把 temporal network 接到 FP 并在 DexYCB 提升，我认为创新论证偏弱；若能证明 outcome-based prior 在相同预算下稳定扩大初始化器输出落入可恢复区域的概率，并带来可重复的闭环/机器人收益，则值得以 RA-L 或 ICRA 为目标。

不建议继续以增加模块数量强化“新颖性”。优先把真实 refiner outcome 的监督优势、梯度可靠性和闭环收益做成一个清楚、可验证的贡献。
