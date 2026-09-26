# V60 — 局部 flow 对照训练完成，几何尚未受益

1000步匹配对照：新增位移/局部相关/真实观测信息后，10°重遮挡真实/代理flow EPE从6.649/6.142降至5.242/5.264px，但XYZ和深度几乎不变；60°对应退化。正确参考位姿仍有约4px不必要位移。未替换默认模型。下一步需要修正零位移行为及大误差粗匹配，并训练flow写入与几何解码的实际耦合。

[完整证据、负结果与回执](../local_flow_v60/SUMMARY.md)。本轮1000步及所有配对评估均已结束，不在后台扩训。

---

# V59 — flow 定位误差拆解完成

256 帧冻结配对诊断：10°重遮挡原可见区域，恢复反馈将 flow EPE 7.896→5.766px，但仍差于零 flow 4.976px；局部头仅把粗匹配5.778px修正到5.766px。60°同类区域71.2%的点超出局部±14px范围。下一步分别验证局部细化的输入/优化能力与大误差粗匹配，不直接放大不准确的flow写入。几何准确目标尚未达到。

[范围、证据与下一步约束](../flow_localization_v59/SUMMARY.md)。本轮没有训练或更改默认模型。

---

# V54 — 全验证集评测完成

40条物理序列、320路相机流、23200帧，三种输入共69600个配对条件。V53重遮挡真实/CAD代理对应XYZ28.996/31.018mm，深度11.717/14.033mm；相对V52只有小幅提升，对应投影误差仍16.324/18.270px。自然可见率<20%子集原始代理深度23.48→23.45mm，几乎没有改善。准确恢复目标尚未达到。578帧不可初始化、215个crop没有GT表面，均单列覆盖率。

[完整结论和图表](../recovery_fullval_v54/full/SUMMARY.md)。全量逐帧文件与终止checkpoint已同步本地并校验，未自动扩训或更改默认模型。

---

# V54 — 正在全验证集评测

对V52 balanced200与V53 final5200作冻结配对比较：40条物理序列、320路相机流、全部23200帧；自然/轻/重遮挡共69600个条件。相同native参考位姿/crop，GT仅评分。初始化失败和空目标区域单列覆盖率，不按预测置信度筛掉误差。7项检查及16帧配对smoke通过；此文字为启动快照，不代表实时进度。

[评测范围与指标](../../../../docs/JEPA_FULL_RECOVERY_VALIDATION_V54.md)。实时远端状态：`unified_jepa_20260921/recovery_fullval_v54_r1/status.json`。

---

# V53 — 新增5000步已完成（2026-09-27核验）

全局5200步，四轮双64帧评估与四次严格续训均完成，当前GPU空闲。相对V52起点，常用重遮挡集真实/代理深度10.424/13.026→8.677/11.351mm，CAD对应XYZ12.278/16.566→11.181/14.146mm。复核重遮挡集对应XYZ11.604/11.725→10.255/9.438mm，但深度11.859/9.653→11.704/9.465mm仅小幅改善；复核全帧真实深度反而回退2.24%。尚不能称为准确几何恢复。

[最终全部里程碑与恢复指标](../recovery_formal_v53/REPORT.md)。终止完整断点保留远端，SHA256为 `717b503de54724e7dad79fe667b9fd49fe392b898af2522267add59ee65bdb3d`；本次进度刷新同步了指标及完成回执，未宣称终止权重已下载。没有自动扩训或替换默认模型。

---

# V53 — 正式 JEPA 恢复训练已启动

按用户“启动正式训练”，从V52 balanced200完整断点接续，新增5000步（全局200→5200）。8 H20、seed42、有效batch32；位姿冻结、历史和DINO特征loss关闭，对应CE权重0.1。保留Adam/EMA/RNG/采样位置，100步平滑回温后缓慢衰减；每50步完整保存，新500/1000/2500/5000步依次评估两套64帧恢复样本。29项启动检查通过。

[正式训练协议](../../../../docs/JEPA_RECOVERY_FORMAL_V53.md)。远端实时状态位于 `unified_jepa_20260921/recovery_formal_v53/status.json`；此本地文字为启动快照，不代表实时步数。

---

# 2026-09-26 — JEPA recovery accuracy is the active priority

用户要求先把 JEPA 恢复做准。V48 位姿读出已停止，完整断点保留在150步；不再安排 PnP／位姿读出训练或评估。

V49 已完成 JEPA-only 固定32训练观测的200步拟合诊断。训练集 real/proxy 深度11.85/11.92→3.78/3.40mm，CAD身份XYZ15.77/14.81→6.04/5.39mm；独立重遮挡反而退化，所以该断点不作为改进模型采用。它只证明恢复链路能拟合，不能证明泛化。

V50 同权重关闭恢复位置RoPE／全部RoPE，原始64样本掩码不变，各几何指标变化不到0.02mm；这不是当前误差的明显开关型根因。

- [当前工作约束与结果](../../../../docs/JEPA_RECOVERY_PRIORITY_20260926.md)
- [V49 完整拟合与独立评估](../recovery_fit_v49/REPORT.md)
- [拟合／泛化对照图](../recovery_fit_v49/fit_vs_generalization.png)
- [V50 几何路由对照](../recovery_routing_v50/REPORT.md)

V51 已完成共享梯度审计：对应 CE 对 patch latent 的梯度约为整体几何监督的7.8–15.3倍；与深度梯度主要接近正交，不能直接称为全面冲突。V52 两组同起点、同数据、各200步，唯一干预为对应权重1.0→0.1；另固定64个新抽样恢复样本复核。两组都已完成。常用重遮挡probe：真实深度12.56→10.42mm，但代理深度12.80→13.03mm、代理CAD对应15.30→16.57mm退化。预先固定的新64样本：真实深度13.28→11.86mm、代理深度11.35→9.65mm，两类对应XYZ均改善。说明监督比例是问题的一部分，尚非一致胜出或准确恢复完成。只评估JEPA恢复，未恢复位姿/PnP任务。

- [V51 实际梯度证据](../recovery_gradients_v51/REPORT.md)
- [V52 两组200步及新64样本复核](../recovery_balance_v52/REPORT.md)
- [V52 恢复指标图](../recovery_balance_v52/paired_recovery.png)

以下保留先前实验记录，均不能替代当前的 JEPA 恢复验收。

---

## 2026-09-26 — V42–V44：对应能力有进展，整体几何仍未达标

完整 8192 点 CAD 读取已实现；查询只来自 JEPA/DPT，输出为实际 CAD 点。修复了离散选点法线梯度异常（约为对应 CE 的 778 倍，首阶段暂关闭该差分项），并验证移除 CE 中坐标先验后，纯学习匹配误差相对配对对照下降约 41%／54%。随后发现最终 DPT XYZ 先验梯度为 0，补齐直接监督并通过 26 项测试。

V44 两组各 100 步已完成；重遮挡真实／代理 CAD XYZ 从对照 14.822／18.763 mm 改善到 14.422／18.336 mm，仍差于原始 source 的 9.482／14.996 mm。没有切换默认模型，没有自动扩训。原始评估标签与 mask 保留。

[V42](../../../../docs/CAD_ATLAS_V42.md) · [V43 配对试验](../../../../docs/ATLAS_DIRECT_V43.md) · [V44 监督修复](../../../../docs/ATLAS_PRIOR_V44.md)

V42/V43/V44 终止断点均已同步并完成本地 SHA256 校验；每阶段 local_checkpoints_verified.json 记录完成回执。

## 2026-09-26 — V41 已完成 100 步；未达标，暂不扩训

监督修复后用相同 V38 raw200 初始化、相同样本和预算完成 V41。原始评估 mask 未变。重遮挡真实区域 CAD XYZ 9.482→9.477 mm，深度 9.871→11.561 mm；代理 XYZ 14.996→12.783 mm，深度 13.473→13.324 mm。相对同预算 V40 并无联合优势，未切换默认模型。

当前视角 CAD 覆盖诊断：自然遮挡目标在 10° 时表面可达率近 100%，90° 时 61.6%，180° 时 47.6%；完整 CAD 8192 点在同阈值下近 100%。阈值为 0.03d，16 帧／10 物体，无模型、无训练。说明小偏差下还存在学习对应不准的问题，大偏差另有单视角表面缺失，不能将失败全部归因于监督噪声。

[本轮实验说明](../../../../docs/GEOMETRY_SUPERVISION_V41.md) · [V41 配对结果](../geometry_supervision_v41/REPORT.md) · [V40 完整 40 序列结果](../visible_correspondence_v40/full_validation/PAIRED_REPORT.md)

## 2026-09-26 — V41 监督独立审计

检查 16 帧／10 个物体，发现并修复 flow teacher 受 BF16 影响的标签量化；新增真实深度／CAD 冲突隔离。15 项测试通过。真实几何监督保留 96.28%，自然遮挡代理与所有原始评估标签保持不变。尚未启动 V41 训练，不把标签筛选统计当作模型提升。

详见 [监督审计与逐帧可视化](../supervision_audit_v41/REPORT.md)。补记：V38 checkpoint 本地交付已完成，189 个文件 SHA 校验通过；下方早先“待下载”是历史状态。

## 2026-09-26: V37–V39 completed; correspondence remains the bottleneck

Goal remains accurate restored geometry, including heavy occlusion. No new model
has met that goal or replaced the default. All GPU runs below have completed.

- V37 frozen-input overfit: one training record reaches XYZ2.313mm/depth1.663mm;
  it has no proxy target, so this is not generalization or CAD-proxy success.
  Eight-record flow-only EPE4.127→1.611px, but holdout5.309→14.732px. Mixed-loss
  EPE ends3.579px train/13.460px holdout. Other/direct gradient ratio50.9x,
  cosine+0.192: scale imbalance, not a proven universal negative conflict.
- V38 matched200-update full-JEPA trials: mandatory CAD lookup, no free XYZ
  offset, stronger flow supervision; compare raw versus CAD canonical targets
  while preserving real depth. The canonical-label change does not solve it.
  Heavy real CAD-XYZ/depth: raw9.482/9.871mm, canonical10.322/10.699mm.
  Proxy XYZ/depth:14.996/13.473mm versus15.165/13.075mm. Same initial tensors
  and optimizer, pose/feature heads unchanged, strict2→200 resume verified.
- V39 same64 heldout cases, same base/crop/masks, frozen original LIP and JEPA,
  history off. Removing only artificial input corruption improves LIP rotation
  from10.556° to5.916° (initial10°). JEPA CAD-flow EPE does not improve:
  real5.625→5.662px, proxy4.809→4.901px, equal physical-sequence mass.
  These are controlled one-step results, not native scores. JEPA depth here is
  decoded depth before the production measurement override; canonical XYZ/flow
  conclusions are independent of that override. Covered-pixel errors must not
  be compared without their different coverage; all-target quality is reported.

Next bounded work should establish visible-surface correspondence with paired
estimated poses of the same observation before heavy reconstruction/history.
A clean-only success would be a prerequisite, not a smaller replacement goal.
No further long run is queued. Preserve all prior pinned runtimes and failed
launch records. Runtime copies are not remote Git checkouts.

Source and reports pushed through a63e6af. V37's40 and V39's68 delivery files
are locally SHA-verified. V38's two complete checkpoints are being downloaded
and will be verified by the active local transfer job; do not claim its local
checkpoint delivery complete until local_delivery_verified.json exists.

See docs/GEOMETRY_LEARNABILITY_V37.md, docs/GEOMETRY_SURFACE_IDENTITY_V38.md,
and docs/GEOMETRY_INPUT_AVAILABILITY_V39.md.

## 2026-09-26: three short geometry iterations completed, goal NOT achieved

V34 paired100-update geometry-only training completed; V35 and V36 each trained
only the new decoder for100 updates with699 existing tensors verified frozen.
All complete-resume checks passed. History and pose/DINO losses remained off.
No official test, multi-seed run, default replacement or long continuation.

Fixed64 training-partition physical holdout, equal physical-sequence heavy mass:

|Model/stage|Real XYZ / depth mm|CAD-proxy XYZ / depth mm|
|---|---:|---:|
|Original V33 source in this protocol|16.651 / 13.821|18.290 / 17.859|
|V34 geometry-only DPT100|15.483 / 13.175|18.863 / 20.470|
|V34 transport100|15.403 / 12.888|18.298 / 19.358|
|V35/V36 changed CAD-mix initialization|13.708 / 10.497|15.283 / 14.191|
|V35 decoder warmup100|13.796 / 10.289|15.996 / 14.870|
|V36 explicit-reference decoder100|13.714 / 10.393|15.636 / 14.842|

The better changed initialization is a CAD-prior mixing effect, NOT a training
or correspondence gain. V36 real/proxy flow EPE5.786/4.874px still loses to
identity5.219/4.463px. All three pilots fail their geometry/correspondence gate.
These controlled10-degree cases are NOT comparable as gains against native or
previous fixed40 baseline-conditioned crops. Detailed configs, preserved failure
logs, metrics and heavy-case arrays accompany each experiment. Do not infer
from these short decoder trials that the entire JEPA latent lacks information.

Next work should establish small-sample correspondence/geometry learnability
before another representation change or long run; cosmetic restoration and
CAD snapping alone do not meet the objective. See docs/GEOMETRY_TRANSPORT_V34.md,
docs/GEOMETRY_TRANSPORT_WARMUP_V35.md and docs/GEOMETRY_TRANSPORT_CONDITIONED_V36.md.

## 2026-09-26: geometry accuracy first, V34 short pilot

User authorized architectural/objective changes and requested small, fast iterations.
V33 completed: full-window vs prefix step2200 native ADD-S@0.05d is
59.009/27.573/13.301% vs59.007/27.018/10.108% (all/<50%/<30%).
The full-window repair did not improve fixed40 geometry: real XYZ39.883mm,
depth15.979mm; proxy XYZ39.573mm,depth23.539mm. No default model promotion.

V34 compares geometry-only DPT with JEPA-predicted CAD-raster transport,
100 updates each, seed42,8H20,batch32. Pose and DINO losses are off;
existing pose heads are frozen. Full-stream single-frame sampling removes
stale earlier-frame bases. The pilot uses64 fixed training-partition physical
holdout records, not native validation/test selection. Input pose perturbation
is controlled10degrees, so these errors must not be compared as gains against
the previous baseline-conditioned fixed40 protocol.

22 targeted tests pass; model/optimizer/RNG resume is checked between stages.
The initial launcher had an EMA API argument error before any completed
checkpoint; the failed artifacts remain under geometry_transport_v34_failed_ema_api.
The corrected run uses a new pinned runtime /tmp/dexycb_geometry_transport_v34_r1.
No long continuation is automatically scheduled. See docs/GEOMETRY_TRANSPORT_V34.md.

> 2026-09-26 V33：完整训练窗口已实测覆盖后段，配对起点和断点恢复通过。新增250步时全帧54.93%、重遮挡22.58%，尚差于同预算前缀对照55.57%/29.51%；仅继续既定500步预算。见 [V33](../../../../docs/FULL_WINDOW_V33.md)。V32已完成，原始配置校验失败的补充验证已修正完成，两个终止权重与494项文件已同步验哈希；[V32结果](../../../../docs/SHARED_PATCH_JOINT_V32.md)。

> 2026-09-26 V32：共享 JEPA patch＋恢复内容联合位姿训练已启动，配对起点与断点恢复验证通过。对照组1450步全帧52.22%，候选尚在训练，未宣称新增路径收益。见 [训练设计与进度](../../../../docs/SHARED_PATCH_JOINT_V32.md)；另发现 [训练时间覆盖与后段退化](../../../../docs/SERIAL_TIME_COVERAGE_AUDIT.md)。

> 2026-09-26 当前进展：V27–V31 已完成实际 latent 读取、像素级观测筛选、几何一致性和补全权重诊断。筛选召回显著提高，但位姿尚未改善；候选均未替换默认模型。见 [实际读取报告](../../../../docs/ACTUAL_READOUT_V27.md)、[观测筛选报告](../../../../docs/DENSE_MEASUREMENT_V29.md)、[几何与权重诊断](../../../../docs/MEASUREMENT_INTEGRITY_V30.md)。主目标仍未达标。

> **最新：已切换JEPA-only训练**。旧联合训练在5900完整断点停止；冻结位姿相关参数、移除pose loss，只训练融合/JEPA/writer及特征和几何恢复，继续剩余4100次更新至10000。异常几何目标过滤已在新阶段启用。27项检查、真实40帧预检和8卡断点恢复通过；见[训练状态与回执](../jepa_only_v11_to10000/REPORT.md)。

> **为何恢复分数好但位姿弱：深入诊断完成**。冻结联合V11/5000，40序列发现特征置换后cosine仍0.756，恢复XYZ60.80mm接近全预测中心61.49mm；10°候选旋转一次修正后仍9.974°/10.003°。FP32下GT隐藏几何只让输出改变0.0053mm/0.001°；补全关系缺少显式XY重投影，固定canonical参考时对相机Z旋转严格不变。梯度平均近正交，未证明普遍负冲突。见[完整机制、控制实验与图表](../recovered_relation_v11_to10000/pose_recovery_mechanism_step5000/REPORT.md)。JEPA-only训练未改动。

> **V11 step5000恢复评估完成**：40序列配对，历史使重遮挡真实特征误差降低14.42%，但空间检索31.3%→27.8%。排除明显异常目标深度后，XYZ仍60.77mm（历史改善0.19%），Depth20.10mm（改善6.39%）。发现正值/有限深度过滤仍会纳入7–16米的错误目标；当时未修改训练，后续JEPA-only阶段已修复。见[恢复、几何与目标质量报告](../recovered_relation_v11_to10000/recovery_step5000/REPORT.md)。

> **V11 step5000与旧LIP同口径对比已完成**：44.10% vs83.66%；GT替换诊断显示全局旋转输出是更大短板，关闭历史仅改善1.16pp，不能解释整体差距。见[完整分组与几何诊断](../recovered_relation_v11_to10000/lip_comparison_step5000/REPORT.md)。训练继续至10000。

> **最新：V11从0训练至10000步**，以V10最新完整4350步权重初始化；旧队列已停止并保留断点。新阶段重建优化器与采样计数，8×H20、seed42，不按指标回退早停。见[正式训练与回执](../recovered_relation_v11_to10000/REPORT.md)。

> **V11恢复几何关系读出已实现并验证**：XYZ/深度/有效性实际参与object query读取，实测优先、补全保留不确定性；21项检查及H20真实40帧/B4编译反传通过。V11尚未正式训练，原V10继续5000步。见[实现与验证](../recovered_relation_v11/REPORT.md)。

> **2500步历史退化调查已深化**：主失败片段的手部深度残差被用于错误修正，历史注入放大过冲；QK并非无效，也未发现重建梯度压倒pose的证据。已完成编译路径闭环干预，当前5000步训练未改动。见[完整诊断、图表与逐帧数据](../v10_supported_history_to5000/deep_investigation_2500/REPORT.md)。

> **最新授权：从历史 writer 修复版 step778 续训至5000步，取消指标早停。** 8×H20、seed42、每50步断点；1000/2500/5000验证后自动继续。此前暂停记录为历史状态，已被本次指令替代。见[续训状态与回执](../v10_supported_history_to5000/REPORT.md)。

> **历史修复验证完成，但未全部验收；正式续训保持暂停**。原step650已保留；修复writer的step778在native重遮挡history-on/off为4.37%/1.90%，但真实特征恢复收益仅0.14%。更密真实patch的step906恢复收益1.54%，仍未达5%，且重遮挡回退，未采用。见[完整根因、配对指标与图表](../v10_history_repair/REPORT.md)。

> **历史修复优先，正式续训暂停**：V10续训已在完整step650断点保留并暂停。发现原writer把前景占格比例当成置信度，导致大量物体历史token被丢弃；已修写入并验证梯度。正在独立128步校准及配对native/history验证，原5000步续训不会自动恢复。见[根因与修复状态](../v10_history_repair/REPORT.md)。

> **V10已授权接续到5000步**：当前step500完整断点已核验，自动接续队列运行中，完成500步验证后继承模型/Adam/RNG/采样状态；1000/2500/5000验证，取消指标回退早停，每50步保存。见[接续计划与回执](../conv_cross_v10_to5000/REPORT.md)。

> **V10 training started**：带state的原全局QK版本，干净seed42、8×H20、有效batch32、500步预算。8卡已核验到step20，断点恢复检查通过；100/250/500步native val，500步history-off。该运行不包含几何历史候选或短实验的writer辅助监督。见[训练状态与固定版本](../conv_cross_v10_state_training/REPORT.md)。

> **V10 几何历史候选（实现及短实验完成）**：36 tests＋H20 40帧/B4 preflight通过；seed42、两臂各64步的两帧对照中，全局QK history-on恢复loss降低9.03%，几何对齐版本−0.04%，未证实位姿收益。未晋升、未启动500步训练，旧V9仍停止。见[完整报告与配对区间](../conv_cross_v10_geohistory/REPORT.md)。

> **2026-09-22 V10 state fusion:** 已将 state token 拼入 RGB→几何 cross-attention 的 K/V；后续仍为共享 patch→JEPA→object query→位姿。28 CPU tests passed（2 CUDA skips），H20 40帧/B4 编译与反传 preflight passed，未启动训练。见 [报告](../conv_cross_v10_state/REPORT.md)。旧 V10 回执保留。

> 最新：V9第2500步评估及冻结latent诊断完成。相对1500步，全帧42.20%→43.04%，重遮挡3.56%→7.32%；主要改善平移和遮挡，旋转关系仍弱。V9保持2500步停训。见 [2500步对比与latent诊断](../v9_step2500_latent_diagnosis/REPORT.md)。

> 同帧双位姿＋零更新验证已完成：三组各400步、单seed42。完整native全帧原模型43.04%，同数据原损失控制35.97%，零更新24.86%，成对＋零更新30.87%。此版本未通过，正式权重保持不变。见 [受控验证报告](../v9_pair_zero_validation/REPORT.md)。

> 旋转路径检查：旋转差异已进入patch，四层JEPA未逐层抹掉响应；object读出方向响应更不均衡，加入GT参考表示后条件可读性提高。详见 [关系编码与旋转辨别能力](../v9_step2500_latent_diagnosis/ROTATION_PATH.md)。主模型保持冻结。

> V10候选结构：观测／渲染配对改为多层卷积＋残差块，外观／几何融合改为cross-attention。29项CPU检查及H20真实40帧零更新检查通过，尚未启动V10训练。见 [实现与验证报告](../conv_cross_v10/REPORT.md)。

# 统一 JEPA 训练加速与断点接续

> 状态更新：下文为此前 v2 的历史加速记录。v2 已停训并保留550步完整断点；新 FP 在线编码＋原 CAD 静态缓存已完成实现与验证，匹配整步为9.61→4.46秒，详见 [加速报告](../fp_staticcad_v3/REPORT.md)。当前训练状态以上方链接为准。

已从第150步完整断点接续，统计时训练到第187步。当前8卡 warmup 的稳定窗口为第166–185步：平均 **9.353秒/步**，原窗口为 **13.058秒/步**；吞吐约 **1.40倍**，每步耗时减少 **28.4%**。

![训练加速与点数诊断](speed_comparison.png)

## 实际改动

- 将 Utonia 的 Hilbert/Morton 整数编码合并为动态长度 Triton kernel；保持原序列次序，避免随点数变化重新编译。
- 使用等价的整数 voxel 分组和 padding，减少逐标量 CPU/GPU 等待。官方安装目录和权重未改动。
- warmup 在前向前冻结 predictor 的参数梯度；保留对输入投影、writer、读出的反传。针对 PyTorch2.8 的 bias-only SDPA 反传问题增加了检查和修复。第500步后恢复联合训练。
- 每帧观测点上限4096→2048；CAD继续8192点。RGB/深度、监督区域、40帧 episode、4帧 TBPTT、有效batch32、学习率和5000步预算保持原配置。

## 速度口径

- 实际训练：8张H20、每卡4条40帧 episode、每步1280帧；旧窗口16–35，新窗口166–185。两窗口均为 warmup，数据批次不同。启动加载、断点写盘和完整验证不计入每步计时。
- 图中 Utonia 微基准为同一GPU、同一批输入的独立编码；不能直接当成整步或8卡速度。
- 第500步后的联合训练仍保留 Utonia 加速；其稳定整步速度要在进入该阶段后测量，不能将当前倍率直接外推到全部5000步。

## 为什么选2048，暂不选1024

第150步固定权重，8条训练 episode，312个计分帧，同一遮挡计划，未进行优化器更新。下表为单GPU每批4条40帧的前向＋反向，未包含DDP/优化器。

| 观测点 | 秒/批 | 训练帧ADD-S@0.05d | 重遮挡平均ADD-S/直径 | 几何特征cosine |
|---:|---:|---:|---:|---:|
| 4096 | 8.552 | 27.244% | 0.19131 | 1.0000 |
| 2048 | 7.993 | 27.244% | 0.21664 | 0.8287 |
| 1024 | 7.599 | 20.833% | 0.22118 | 0.6252 |

2048在该小样本的命中率未变，但重遮挡平均误差增加约13.2%，因此减点不属于无损优化，也未证明验证集精度保持。1024的命中率下降约6.41个百分点，未用于正式接续。几何cosine来自各自闭环预测裁剪，包含采样及随后的位姿变化，不能解释为纯编码器误差。上述诊断使用候选缓存；正式接续另将原CAD特征按原值迁移。

## 验证与状态保留

- 44项回归通过、1项跳过；真实RGB-D预检验证监督区域、同一patch梯度、writer后续帧梯度、冻结编码器、不传入GT teacher和无深度路径。
- 16种整数深度×4种序列编码、padding边界和五批观测点特征对照通过；所测旧/新Utonia特征逐值一致。
- 20个CAD的原缓存与旧/新编码器即时输出逐值一致。原始特征文件按原值迁移到绑定新源码的缓存版本，20个文件SHA保持原值。中间重新生成的缓存有3个物体出现最大0.000955差异，原因未进一步确定；它们已归档，不进入正式接续。
- 8卡断点恢复前的模型/优化器/调度器/RNG逐值相同；重放后模型最大差3.38e-06、优化器最大差0.000168，在预设1e-5/1e-3容差内，**不宣称下一更新跨进程逐位相同**。同进程 warmup→joint 转换也通过。
- native接口24帧、重建probe100行、6组teacher图以及4×40帧异常深度回放通过。完整native验证仍在500/1000/2500/5000步执行；原停止条件不变。
- 原第150步权重、优化器、调度器、8份RNG和sampler位置4800全部保留；旧源码/日志/完整checkpoint已归档。仅重算未写断点的151–157步。新模型默认项未替换，没有多seed或official test。

## 回执

- 源码 SHA256：`46a516f714807c1ba786c09f15ec220c377d41a88706e125dccdfdc5c048b7c0`。
- 原 step150 checkpoint：`c29bd065ca242c48337175b5721e1df4a3041a085a8932146e799cf3ffe9db87`。
- 保留训练状态后的迁移 checkpoint：`0e5a99ecbdb4349fe6269b470e5f427e6a43b07c961d9ee5e687b7ba67f7a44f`。
- [实际速度数据](live_speed.json)、[配对点数CSV](paired_point_budgets.csv)、[本地/远端源码校验](source_sync_receipt.json)。
- [状态迁移](../artifacts/formal_rgbd_v2/runs/seed42_missing_only/speed_migration.json)、[8卡回执](../artifacts/formal_rgbd_v2/preflight/ddp_speed/receipt.json)、[CAD迁移](../artifacts/formal_rgbd_v2/speed_audit/cad_canonical_carry.json)。
- 本地和远端同步的是运行源码及实验产物，不表示进行了Git提交或对齐Git历史。


## 2026-09-23：恢复能力优先，训练已停止

按用户要求停止JEPA-only，完整断点保留在7400步（8卡RNG及optimizer/scheduler齐全）。当前正式训练保持停止。空间特征／几何一致性／有真实历史支持的补全监督已在独立版本实现，完成零更新检查及40序列三路历史诊断；未宣称训练后提升。详见[恢复优先报告](../jepa_recovery_focus_v1/REPORT.md)。


## 2026-09-23：JEPA 恢复训练已启动

用户指令“训jepa”后，已启动空间／几何／历史恢复目标，7400→10000，8×H20，位姿相关模块冻结。7500/8000/10000进行40序列三路历史恢复评估。[训练状态](../jepa_recovery_focus_v1/TRAINING.md)。


## CAD局部表面V12已接续训练

旧恢复版本在8850完整断点保留。新V12从8850接续到原预算10000，新增局部Utonia表面读取与对应监督，MLP保留、DPT暂未加入。30项测试、H20预检及8卡8853启动/恢复核验通过。[V12报告](../cad_surface_jepa_v12/REPORT.md)。


## Trainable DINO + EMA, then no-history frame batching

V12 preserved at9750; trainable-DINO/EMA V13 ran to9800 with exact startup/resume audits. User then disabled history. Eight-frame batching passed matched gradient/crop checks and improved no-history throughput1.267x (6.3955→5.0478s). Continuing9800→10000 with full optimizer/EMA state. Details: [EMA](../cad_surface_ema_v13/REPORT.md), [no-history batching](../cad_surface_ema_v13_speed/REPORT.md).


## V17 DPT / 3D RoPE / 法线监督

新实验已从step23400完整断点接替V16，新增1000步至24400。详情见 [V17报告](../dpt_rope3d_v17/REPORT.md) 与 [实时进度](../dpt_rope3d_v17/LIVE_STATUS.md)。


## V18 分阶段 RoPE

已从step24400接续1000步至25400，保留DPT和Adam，实测优先、恢复表面供后段CAD读取。[V18报告](../staged_rope_v18/REPORT.md) · [实时进度](../staged_rope_v18/LIVE_STATUS.md)。

## 2026-09-27：V56 flow／JEPA 双向交互原型

从V53完整模型权重完成200步恢复训练。恢复XYZ帮助flow定位：人工／自然遮挡
目标EPE分别降低30.6%／34.9%（同一权重开关反馈）；反向flow写入对几何影响
仍很弱，重遮挡自然代理深度7.732→9.052mm，未通过。预算结束，未替换默认模型。
这是受控留出样本诊断，不是native全评测。[结论与消融](../flow_reconstruction_v56/SUMMARY.md)。

## V57：冻结几何信息与监督一致性审计

预测flow与恢复深度联合做几何拟合，在10度参考误差的重遮挡CAD代理上，XYZ
8.846→6.616mm、深度9.052→7.426mm；真实深度及60度对照仍未通过，未替换主模型。
点可见性AUROC仅0.624，0.5阈值会丢弃全部测量锚点。直接GT CAD与真实深度
仍有约10mm差异，已检查固定样本的原始标注、mesh中心和像素投影，成因未确定。
[完整结果与监督图](../flow_surface_audit_v57/SUMMARY.md)。
