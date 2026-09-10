# Codex 实施任务：DexYCB-LIP v1

> 工程代号：Latent Interaction Pose Tracker v1。本文件是新模型的实施规格，不是现有论文的复现声明，也不表示模型已经训练完成。
> 目标环境：用户提供的单机 8 × NVIDIA H20；实际显存、驱动、可用 CPU 和磁盘以本机检测为准。
> 第一目标：产出一个能独立训练和闭环评估的、无显式手建模的 RGB-D 时序物体位姿模型。FoundationPose 集成是独立可选模块，不得阻塞主模型训练。

## 0. 执行方式与边界

请实际创建代码、配置、测试和可运行命令，不要只输出架构建议，不要留下主路径上的 TODO、假函数、随机预测或伪训练结果。

先阅读当前仓库的 AGENTS.md 和现有代码，避免覆盖用户已有实现。只修改工作目录；原始数据集只读。依赖使用隔离环境，不升级宿主机驱动、不卸载用户的 CUDA/PyTorch、不删除既有 checkpoints。不申请云计算资源，不自动下载整个 DexYCB，也不自动接受任何数据许可。

依次完成：

1. 环境检查与接口核对。
2. 数据读取、坐标/渲染验证、单元测试。
3. 主模型实现、合成数据反向传播测试、真实小样本过拟合测试。
4. 单卡显存/吞吐预检与 8 卡 DDP 短跑。
5. 给出通过预检的正式训练命令；提供独立的前台执行脚本。

默认 Codex 的实施会话只执行测试和短跑，不自动启动 40,000-step 长训练。只有用户额外明确要求直接启动长训练时才启动，必须记录真实进程、日志与 checkpoint 状态。不要把“启动了任务”说成“已经训完”。遇到数据、网络权限或 GPU 访问阻塞时，完成其他可完成部分并明确报告，不伪造通过。

资源约定：

- `DEX_YCB_DIR`：已解压的 DexYCB 根目录，必填。
- `LIP_WORK_DIR`：当前项目根目录，默认当前目录。
- 运行输出默认 `./runs/`；缓存默认 `./cache/`，不得写入原始数据目录。
- `FOUNDATIONPOSE_DIR`：可选的 FoundationPose checkout。
- 预训练 RGB 权重优先读取本机缓存；需要联网下载时遵循运行环境的审批策略。

## 1. 模型问题定义

不训练手关节、手掌、MANO、手 mesh、接触点或手分割。前向过程不接收上述标注，也不把它们用于辅助监督。

目标模型：

`unmasked RGB-D history + CAD geometry rendered at a previous estimated pose + past estimated object poses -> latent z -> current object-pose correction`

输入包含当前帧 I_t / D_t，因此严格地说这是 interaction-conditioned pose updater / initializer，不是只使用过去数据的纯动力学 prior。时序必须因果：允许当前帧，不允许未来帧。

第一版只实现确定性位姿头。暂不增加 diffusion、VAE、多假设、未校准 covariance gating、显式接触模型或端到端 FoundationPose 训练。latent 是否利用了交互上下文，需要实验验证，不能由结构名称直接宣称。

## 2. 需要核对的上游契约

以本机实际 checkout 为准，记录使用的 commit hash 和版本。参考来源列在文件末尾。

DexYCB 官方 reader/说明包含 color、aligned depth、相机内参、`meta.yml` 和 `labels_*.npz`；物体 pose 位于 `pose_y`，不是手的 `pose_m`。[S1, S2]

FoundationPose 的 `pose_last` 是针对居中 mesh 的内部位姿，公开返回值还乘了 `get_tf_to_centered_mesh()`。不得把公开返回值不经转换直接塞回 `pose_last`。[S3]

不要为了读取 DexYCB 而引入 MANO 运行依赖。允许独立实现最小 reader，严格对齐官方 split 与字段含义；如复用第三方源码，保留许可与出处。

## 3. 坐标、单位和状态约定

所有几何计算统一使用米、弧度、OpenCV optical camera axes：x 右、y 下、z 前。

外部位姿 `T_original = ^C T_O` 把原始 CAD 坐标的点变到相机坐标。采用列向量和左乘矩阵，不混用 camera-to-object。

对原始 mesh 顶点 v：

```text
c = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
C = Translation(-c)                    # original mesh -> centered mesh
v_centered = v_original - c
T_centered = T_original @ inverse(C)
T_original = T_centered @ C
```

模型内部状态与训练统一使用 `T_centered`，公开 API 同时返回 original 与 centered，字段名明确标注。验证两套坐标投影完全一致。

尺度 d 使用 mesh 的直径（米）。在预处理中通过 convex-hull vertices 的分块最远点距离计算并缓存；不要把 bbox 对角线暗称为模型直径，不要在每个 batch 重算。不能擅自把 CAD 缩放到单位球后仍使用原始 pose。

深度读取必须有显式的 `depth_scale_to_m`。先检查数据发行版/官方读取实现/标定；若使用 0.001 作为初始配置，必须在数据审计中注明并通过物体可见区域的渲染深度一致性检查，不能把所有数组一律除以 1000。`pose_y` 和 CAD 的单位单独核对。

渲染器如内部使用 OpenGL/NDC，只在 renderer 边界做转换，模型状态仍用 OpenCV。测试相机前方、图像上下方向、深度值和投影的一致性。

## 4. 数据集与采样

### 4.1 Reader 与 split

默认使用官方 `s0_train / s0_val / s0_test` 定义，不随机切帧。[S1]

最小记录：

```text
subject_id, sequence_id, camera_serial, frame_index, timestamp
rgb_path, aligned_depth_path, intrinsics
object_id, object_index_in_sequence, pose_original_gt
mesh_path, mesh_center, mesh_diameter
```

正确选出被抓目标：

```python
j = int(meta['ycb_grasp_ind'])
object_id = int(meta['ycb_ids'][j])
pose_gt_3x4 = labels['pose_y'][j]
```

不能把 `ycb_grasp_ind` 当作 YCB class ID，也不能用 `object_id - 1` 直接索引 `pose_y`。`pose_y` 通常是 `[num_obj, 3, 4]`，补齐齐次行。[S1, S2]

只跟踪该序列的 grasp target，不把桌上所有静止物体当作主要训练样本。空 pose、损坏文件、非有限矩阵、非法 R、缺帧必须审计并计数；不能静默替换成 identity GT。

一个样本只能来自同一 `subject + sequence + camera_serial + target object`。不同同步相机流可各自训练，但不在单个样本里做多视角融合。s0 的同一真实录制 sequence 不能跨 train/val/test。输出 split manifest 与互斥检查结果。

为后续单独重训预留 s1 配置，但不要把 s0 模型在它训练见过的 s1 subject 上评估并宣称 unseen-subject generalization。第一版不要求跑 s2/s3。

### 4.2 时序窗口

默认 L=8，输入时刻 `t-7s, ..., t-s, t`，训练 stride s 从 {1,2,4} 采样，概率 {0.6,0.3,0.1}。

使用真实时间戳；若发行版只有固定帧号，则显式配置并核对 fps，时间位置编码使用实际 frame difference / fps，不能把 stride=4 当成 stride=1。

每个目标窗口允许读取 GT 作为监督，但特征构造只能使用早于 t 的历史估计、I_<=t 和 D_<=t。当前 T_t_GT 不得进入 crop、输入渲染、数值状态 token 或任何 forward 参数。

训练的 20% 样本随机采用较短有效历史（1/2/4 帧或可用长度），左侧 padding 并提供 valid mask，以覆盖序列刚开始时的冷启动。padding 帧不是新的真实观测，不得产生 NaN attention。

### 4.3 避免静止样本主导

先在 train split 统计目标运动和可见性 proxy。使用固定混合采样：50% moving clips、25% heavy-occlusion clips、25% uniform clips；桶之间允许交叠。桶内先均衡 target object / physical sequence，再采 camera 和时间，防止某几个长序列主导。

初始 moving 定义：窗口内相对首帧的中心位移超过 0.05*d，或转角超过 5 度；阈值可配置。heavy-occlusion 定义使用下面的 visibility proxy < 0.3。缺少某个桶时显式报告并重新归一化，不能无限重采样。

`seg == object_id` 仅允许用于离线 train 采样统计、训练遮挡增强的位置、第一帧初始化 mask 的独立实验，以及评估分层。绝不作为主模型的输入通道，也不用于 t>0 的真实跟踪 crop。

visibility proxy：目标 GT pose 单独渲染的 silhouette 为 M_gt；V=(seg==object_id)。

`visibility = |V intersect M_gt| / |M_gt|`

这是标注/渲染得到的 proxy，不是精确可观测性。M_gt 为空时记 unknown，不能丢弃该帧的其他 tracking 指标。禁止读取/利用手的 255 类来训练手分割。

### 4.4 IO

先实现 raw reader + 小型 metadata/pose manifest cache；缓存中记录源路径、帧索引、版本、单位与 mesh checksum。

不要离线保存每个重叠 8-frame clip；这会大量重复数据。允许后续按 sequence/chunk 建立只存一份 frame 的缓存，但不得成为第一版运行的必要条件。

DataLoader workers 中不得创建 CUDA renderer。GPU rendering 在各 DDP 主进程的 local device 上完成。提供 data_time / render_time / model_time 分离 profiling，不能把 GPU 低利用率默认归咎于模型太小。

## 5. 因果 crop、CAD render 与几何输入

每个预测时刻，令 `T_base` 为最近一次已接受的物体估计（内部 centered 坐标）；默认是上一采样时刻的估计，不是当前 GT。

将 mesh 用 T_base 投影，构造 square ROI，默认边长为 projected bbox 最大边的 2.0 倍，最小 64 原图像素，越界 padding。训练时可从 [1.6,2.4] 随机选扩张比例。整个 8-frame 窗口使用同一个 ROI，不要逐帧用 GT bbox 重新居中。

crop/resize 后必须更新相机内参；通过像素中心约定明确的变换 A 实现 `K_crop = A @ K`，并通过投影测试。保留 A 供 debug 和数值 conditioning 使用。RGB 用双线性采样；depth 与 validity 使用不跨越前景边缘制造伪深度的策略，第一版可用 nearest。

RGB 保留完整 ROI，包括手、背景和其他物体。禁止乘 object segmentation 把交互上下文抹掉。

只在 T_base 渲染一次 CAD 几何，供此窗口的所有观测帧复用。不要用当前 T_GT render 作为输入。第一版只渲染 depth、silhouette、canonical object XYZ，不依赖纹理渲染。

推荐 CUDA rasterizer `nvdiffrast.RasterizeCudaContext`，无桌面显示依赖。[S5] 不需要 renderer 对 pose 求梯度。安装不可用时可以保留单独 CPU correctness backend，但不得用全零 render 伪装完整模型。

令 z_base 是 T_base 的相机 z，d 是 mesh diameter。每帧 geometry 输入严格为 9 通道：

```text
0: (D_obs - z_base) / d
1: valid_depth
2: (D_render - z_base) / d
3: render_silhouette
4..6: X_object_centered_render / d
7: (D_obs - D_render) / d
8: valid_depth & render_silhouette
```

所有无效值在归一化后置零，并保留对应 mask。深度/残差值可配置 clip 到 [-2,2]，XYZ 不随意改单位。深度残差不等于手遮挡标签：它也可能来自初值误差、物体运动和深度噪声。

输入接口建议：

```text
rgb                  [B,L,3,224,224]
geometry             [B,L,9,224,224]
history_state        [B,L,state_dim]
base_state           [B,base_dim]
time_offsets_sec     [B,L]
frame_valid          [B,L]
history_pose_valid   [B,L]
T_base_centered      [B,4,4]
object_diameter_m    [B]
```

history_state 对历史 j<t 编码：R_j R_base^T 的前两列（6D rotation representation）、(t_j-t_base)/d、pose valid。当前时刻没有已知 pose，填零并标为 unavailable。base_state 编码 R_base 前两列、t_base/d、log(d/1m)、K_crop 的 fx/fy/cx/cy 除以224。不要把 object ID、subject ID 或 camera serial 当作模型的可学习类别 embedding。

## 6. 主网络：固定第一版结构

### 6.1 RGB encoder

使用 torchvision ResNet-50 的 ImageNet 预训练权重，取到 layer3 的 stride-16 特征，不运行 layer4/fc。[S4]

```text
[B*L,3,224,224]
 -> ResNet50 stem + layer1 + layer2 + layer3
 -> [B*L,1024,14,14]
 -> 1x1 projection
 -> [B*L,256,14,14]
```

固定 BatchNorm running statistics，包括调用 model.train() 后也要保持冻结状态。不要用含未来帧统计的 batch-dependent normalization 破坏在线一致性。权重实际选择与来源写入 checkpoint/config。

### 6.2 Geometry encoder

独立轻量 CNN，从9通道输入降采样到 [B*L,256,14,14]：四级 stride-2 Conv，channels [32,64,128,256]，每级 Conv/GroupNorm/GELU + 一个同通道 residual block。使用有效的 GroupNorm group count，不依赖大 batch statistics。

### 6.3 Spatial interaction fusion

RGB 与 geometry 都 flatten 为196个 token，加相同坐标约定的2D位置编码。

用2个 cross-attention block：RGB tokens作 Q，geometry tokens作 K/V；d_model=256，8 heads，FFN=1024，dropout=0.1，pre-LN、residual。不能先 global-average-pool 再融合。

每帧空间融合结果 reshape 为14x14，再 adaptive-average-pool 到4x4，保留16个 spatial tokens。另加一个由 history_state 和 base_state MLP 编码的 state token，每帧17个 token。

### 6.4 Temporal latent encoder

对 L*17 个 token 使用4层 temporal Transformer：d_model=256、heads=8、FFN=1024、dropout=0.1、pre-LN。

加 actual time-offset encoding、spatial position encoding、token-type encoding。按帧建立 causal mask：同帧 token 可互相看，不能看后面的帧。另加一个位于最新时刻的 learnable readout token，可看所有有效历史与当前 token。

最终 readout 为 z_t∈R^256。整个网络不得输出或中间监督手的骨架、mesh、MANO 或 hand mask。

### 6.5 位姿修正头

MLP：256 -> 256 -> 6，GELU。最后一层权重和bias初始化为0，从 identity correction 起步。

输出分为两个3维向量：

```text
delta_rotvec      radians, camera-axis relative rotation
delta_center_norm camera-axis center displacement / object diameter
```

更新规则：

```text
R_pred = Exp_SO3(delta_rotvec) @ R_base
t_pred = t_base + d * delta_center_norm
```

这是解耦的 SO(3) rotation + camera-frame center translation，不是直接对 `[delta_rotvec,delta_t]` 做左乘 SE(3) exponential。不要旋转 t_base 导致绕相机原点转动，不要混淆米和无量纲位移。

所有 pose update、SO(3)运算、矩阵检查和 loss 在 FP32 中计算。几何单元测试可用FP64。

模型forward返回 `pose_centered`, `pose_original`, `delta_rotvec`, `delta_center_norm`, `latent`。第一版不把 latent 转换成 hand state，也不输出未经校准的“概率置信度”。

## 7. 监督目标与 loss

有噪声或 rollout 的 T_base 时，标签必须相对该 T_base 重新计算：

```text
delta_R_gt = R_gt @ R_base.T
delta_center_norm_gt = (t_gt - t_base) / d
```

绝不能仍用 clean GT(t-1) -> GT(t) 的 delta 监督 noisy base 的修正。

默认损失：

```text
L_t   = SmoothL1((t_pred - t_gt)/d, beta=0.02), coordinate mean
L_R   = geodesic_angle(R_pred @ R_gt.T), radians
L_pts = mean || T_pred X_i - T_gt X_i ||_2 / d
L     = 1.0*L_t + 0.5*L_R + 1.0*L_pts
```

X_i 为固定随机种子生成的512个 centered CAD surface points，各物体缓存，不必对每个 batch 重采样。loss 归一化必须明确到 sample，再按有效 sample 平均。

SO(3) Exp/Log 和 angle 实现需对零旋转、小角度、接近π验证数值稳定与梯度。不要用会在 identity 产生 NaN 的裸 acos 公式。

第一版默认 canonical GT 监督，明确记录这个选择。评估同时报告 ADD 和 ADD-S，不能只用 ADD-S 掩盖对称物体的旋转翻转。可扩展经过核对的 symmetry metadata，但不得按物体名称猜测对称性，也不能把某个几何近似对称当成完整纹理对称。若加入对称处理，所有 pose、mesh frame 和 equivalent GT 的选择必须一致。

不要默认添加趋向零速度/零加速度的平滑 loss。快速真实动作不能被当作噪声。以后增加 temporal loss 时应监督 GT-relative motion consistency，而不是强迫静止。

## 8. 抗 exposure bias 训练

### 8.1 历史位姿扰动

训练不得只输入完美历史 GT。warmup/主阶段使用 GT history + 有时间相关性的估计误差。

建议初始 mixture（均可配置）：

```text
75% small:  rotation per-axis std 2deg;  center per-axis std 0.01*d
20% medium: rotation per-axis std 8deg;  center per-axis std 0.05*d
 5% large:  rotation per-axis std 20deg; center per-axis std 0.10*d
```

误差使用 AR(1), rho=0.9，并有窗口共享bias。先定义要保持的边际标准差再用 sqrt(1-rho^2) 缩放创新噪声，不能让方差无意累积爆炸。作为初始化，角度误差范数最大45度，中心位移范数最大0.25*d；记录 clipping 比例。

扰动后重新生成 T_base、crop 和 render，并重新计算 target correction。采用概率0.2的 past pose-token dropout，但 T_base 本身仍存在。

### 8.2 图像/深度增强

窗口内一致的轻量颜色变化，轻量深度噪声/洞、运动模糊。不要无几何补偿地水平翻转 RGB-D；不要独立抖动每一帧 crop。

先保留真实遮挡数据作为主信号。训练期可以增加目标区域内的观测擦除：利用 train target mask 定位，但不给模型该 mask；只修改 observed RGB/depth，并正确更新 depth validity，保留周围上下文。采用随机不规则区域，不能把GT silhouette边界以擦除形状编码给网络。不要擦掉 CAD render；物体先验本来就是可用输入。

合成擦除不等同真实物理手遮挡，必须有关闭此增强的开关，val/test不使用。

### 8.3 短闭环 rollout

后期训练一半 batch 使用 U=4 的短 rollout，另一半仍为单步。各rank用同一个global-step种子决定 U，保证分布式 backward/sync 次数一致。

rollout起点的历史允许用 noisy GT 初始化，之后每一步把模型自己的输出写入历史，再推进下一帧，并重新算 crop/render。不能每步偷偷恢复 GT。

每一步预测用于下一步状态时 detach；每步 pose loss 都对当前 forward 反传，按 U 归一化。可以逐步 backward 回收图，避免把4步全部图堆在显存里。这是截断的闭环训练，不声称对整个 tracking/rendering 过程做全程可微优化。

多步样本需要连续的 L+U-1 个采样时刻，不能越过真实sequence或缺帧。与8-frame inference保持一致。

## 9. 单机8卡训练配置

使用 PyTorch DDP + torchrun + NCCL，AMP bf16。DDP/AMP 与 torchrun 用法遵循当前实际安装版本的官方API。[S6, S7]

以下是起始超参数，不是已经实测的最优值：

```yaml
seed: 42
image_size: 224
clip_length: 8
world_size: 8
batch_size_per_gpu: 16
grad_accum_steps: 2          # effective global clip batch = 256
precision: bf16
optimizer: adamw
lr_new_modules: 0.0001
lr_rgb_backbone: 0.00001
weight_decay: 0.05
no_weight_decay_for: [bias, norm]
max_optimizer_steps: 40000
warmup_optimizer_steps: 2000
scheduler: cosine
min_lr_ratio: 0.1
grad_clip_norm: 1.0
num_workers_per_rank: 4
prefetch_factor: 2
pin_memory: true
persistent_workers: true
save_every_optimizer_steps: 1000
quick_val_every_optimizer_steps: 2000
full_val_every_optimizer_steps: 5000
```

阶段：

- step 0–1999：RGB backbone 的学习率为0；训练 geometry/fusion/temporal/head。
- step 2000–29999：RGB backbone LR逐步升到1e-5，新模块峰值1e-4；单步噪声历史训练。
- step 30000–39999：50%单步 + 50% U=4自回归训练；沿用已经衰减的学习率，不突然重启大LR。

初期“冻结”可以仅用lr=0实现而不改变requires_grad，以避免DDP参数注册在解冻时出错。必须冻结BN running stats。scheduler按optimizer step推进，不按microbatch推进。

启动前打印8张卡的实际型号/显存、driver、PyTorch、CUDA runtime、bf16支持、CPU/磁盘可用情况。不要假设所有H20发行版规格一样。不能虚报FPS、显存或训练时长。

提供 `tools/probe_batch.py`，先单卡测试 {4,8,16,32} clips，包括forward/backward和U=4路径，选择显存安全的配置，默认不超过探测时可用显存的80%。对应调整accum，使有效global batch仍为256；选择结果写入resolved YAML。不要在某个rank OOM后单独修改batch继续跑。

所有geometry/renderer/loss关闭autocast，网络主体bf16。默认不使用FP8。若bf16不可用则明确报错或显式选择FP32，不静默切换FP16。

每个rank设置local CUDA device；每rank一个renderer context。gradient accumulation使用正确的DDP no_sync，forward也在no_sync上下文中。rollout的中间backward不得意外触发不匹配的all-reduce。

sampler必须DDP-safe，所有rank具有相同step数；不要简单叠加WeightedRandomSampler和DistributedSampler。允许按global_step/rank/microstep确定性采样；保存随机种子与采样状态。

checkpoint保存model/optimizer/scheduler/global step/config/mesh metadata hash/split hash/RNG state；写文件使用atomic rename。支持恢复训练，resume不能从头开始LR schedule。

默认日志本地JSONL + TensorBoard，远程logging须用户启用。分别记录data/render/forward/backward time、clips/s、observed frames/s、samples seen、显存峰值、有效batch、旋转与平移loss、更新幅度、非有限值计数。torch.compile为可选项，不能成为成功运行的前提。

## 10. 评估协议

必须同时实现两种评估，名字分清：

### A. One-step diagnostic

统一的noisy历史/初值条件下预测下一帧，比较：

- Zero-motion：直接返回T_base。
- Constant-velocity：用最近两个可用估计的camera-frame rotation增量和center displacement，按实际时间间隔外推；不足两帧退回zero-motion。
- Learned updater：本模型。

所有方法使用同一批初值与时间间隔。保存seed/manifest，不挑选模型有利的初始化。

### B. Full-sequence closed loop（主要结果）

从每个sequence-camera stream的首个有效pose初始化一次；之后逐帧使用自己的历史输出，不能再读GT进state/crop/render。报告 `initial_pose_source=gt_first_frame`，这不是端到端检测评估。

刚开始历史不足时使用有效历史mask，不借用未来帧。跟丢后不准用GT重置，也不剔除跟丢片段。可预测固定last-valid state并标记lost，但仍计算误差。GT和目标seg仅在独立metric端使用。

注意：在“只有首帧GT、之后没有观测更新”的完整序列里，纯constant-velocity无法凭空获得初始速度，可能退化为zero-motion。不能靠击败这样的基线宣称交互建模有效。恒速的主要对比放在A的相同历史条件下，以及后续相同FP观测更新的组合实验中；时序模块的真实收益还需要与单帧视觉模型比较。

评估输出：

- camera-frame center error：mean/median/p95，mm；original CAD origin error可作为额外字段，不能混淆。
- rotation geodesic：mean/median/p95，deg。
- ADD、ADD-S（米），以及各自低于0.05d / 0.1d的比例。
- per-object、per-sequence和macro/micro平均。
- visibility bins：[0,0.1)、[0.1,0.3)、[0.3,0.6)、[0.6,1]，另列unknown。
- moving vs non-moving子集，速度阈值在train上确定并固定。
- 首次连续5帧ADD-S > 0.1d的时刻/帧数，lost fraction，不能称为普适工业容差。
- batch=1在线延迟，分preprocess/render/network/optional FP和end-to-end，CUDA计时必须同步。

快速val用固定manifest子集，完整val才选择best checkpoint。主选择指标为full-sequence macro ADD-S recall@0.1d，必须同时展示ADD和rotation以暴露对称翻转。test只在模型/超参数确定后运行，不用test选择checkpoint。

另提供single-frame配置（同一个网络，L=1，训练条件一致）和reduced-context配置作为后续ablation；默认不自动启动多组完整训练。时间打乱/遮蔽上下文可以用于诊断，不作为证明因果交互理解的唯一证据。

## 11. FoundationPose adapter（可选，不阻塞主模型）

不改FoundationPose backbone，不把 `.predict()` 当成端到端可微训练接口。

主模型给出external original-mesh pose之后，按照实际FP代码转换：[S3]

```python
C_fp = fp.get_tf_to_centered_mesh()         # original -> FP-centered
T_prior_centered_fp = T_prior_original @ inverse(C_fp)
fp.pose_last = as_float32_tensor_on_fp_device(T_prior_centered_fp)
T_refined_original = fp.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
```

上面仅为数学接口约定，实际实现要核对shape、dtype、device以及FP版本的函数签名。测试使用非零mesh center，验证round trip和投影一致。

若不接受某个refinement结果，FP内部pose_last也必须恢复为实际采纳的状态，不能外部输出prior而内部继续保留已拒绝结果。第一版不增加凭空假定已校准的gate。

可选对比：vanilla FP / CV initializer + FP / learned initializer + FP。相同第一帧初始化、相同refine iteration数、相同帧和评估协议。没有FP依赖或权重时明确skip真实FP测试，不得伪造成功或因此阻塞主模型。

## 12. 建议项目结构与必须实现的CLI

结构可在保持清晰接口的前提下调整：

```text
pyproject.toml
README.md
configs/
  dexycb_lip_v1.yaml
  dexycb_lip_v1_single_frame.yaml
  smoke.yaml
src/lip/
  data/                    # raw reader, official split, clip sampler, transforms
  geometry/                # so3, frames, crop, renderer, mesh metadata
  models/                  # RGB/geometry encoder, fusion, temporal, pose head
  losses/
  engine/                  # DDP, train loop, rollout, checkpoint
  evaluation/
  integrations/foundationpose.py
  train.py
  evaluate.py
tools/
  audit_data.py
  build_index.py
  check_geometry.py
  overfit_small.py
  probe_batch.py
scripts/
  smoke_8gpu.sh
  train_8gpu.sh
tests/
```

至少提供下面这些真实可运行的入口，README与实际argparse一致：

```bash
python tools/audit_data.py --data-root "$DEX_YCB_DIR" --out runs/audit
python tools/build_index.py --data-root "$DEX_YCB_DIR" --setup s0 --out cache/dexycb_s0
python tools/check_geometry.py --data-root "$DEX_YCB_DIR" --index cache/dexycb_s0 --num-samples 32 --out runs/geometry
pytest -q
python tools/overfit_small.py --config configs/dexycb_lip_v1.yaml --num-clips 32 --steps 500
python tools/probe_batch.py --config configs/dexycb_lip_v1.yaml --out configs/resolved_8gpu.yaml
bash scripts/smoke_8gpu.sh
bash scripts/train_8gpu.sh
```

train_8gpu.sh默认在前台运行，不用nohup、隐藏后台任务或自动提交集群作业。核心启动命令：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m lip.train \
  --config configs/resolved_8gpu.yaml \
  --data-root "$DEX_YCB_DIR" \
  --index-root cache/dexycb_s0 \
  --output runs/lip_v1_s0
```

支持 `--max-steps` 短跑与 `--resume`。提供一个一键preflight模式，但不能绕过数据/几何校验。

## 13. 必须通过的测试与交付验收

1. object index mapping：包含多个YCB对象、grasp target不在第一个位置的样本。
2. original/centered round trip：非零mesh center、相同3D点投影，数值误差在设定容差内。
3. crop intrinsics：crop前后投影映射一致，正确处理padding与pixel center。
4. identity与已知pose delta的update：纯平移不改变R，纯旋转不绕相机原点移动中心。
5. SO(3)近零/近π、梯度有限、R正交、det(R)=+1。
6. 全零depth、部分invalid、全遮挡、padding历史、短序列不会产生NaN。
7. s0 split物理sequence不相交，clip不跨camera/sequence。
8. causality：替换未来帧/未来GT后当前预测不变；更改当前GT但不改图像与history时forward输入/输出不变。
9. 无hand annotation测试：移除MANO、hand joints、pose_m等字段后仍能完成主模型训练。
10. 合成数据CPU前向/反向；没有CUDA renderer时只跳过明确标记的GPU测试，不能伪造render。
11. 真实32-clips过拟合：固定采样、关闭随机增强/噪声，记录loss下降及相对zero-motion baseline的误差变化。若失败，先检查几何/target/梯度，不直接盲目增加模型大小。
12. 8卡DDP 50 optimizer steps + U=4 rollout短跑，无hang、无NaN、各rank样本与step数一致。
13. 保存/恢复后继续若干step，global step与scheduler一致；记录非确定性容差而不是虚假bitwise保证。
14. 闭环评估无GT状态重置，lost帧不消失，指标与可视化同一份预测。
15. optional FP adapter的非零mesh-center测试；真实依赖未提供时清晰skip。

必须交付：

- 可运行主模型、数据/训练/evaluation代码和配置。
- 实际环境依赖锁定记录，不编造未安装的版本。
- 含图像叠加、GT/render深度核对、ROI与中心坐标的geometry审计产物。
- 小样本过拟合日志、8卡短跑日志、显存和吞吐probe结果。
- best/last checkpoint命名约定、resume与eval命令。
- 一份事实性状态报告：哪些已实现、哪些实际运行通过、哪些由于外部条件未执行。所有数字必须来自真实日志。

实现顺序优先主路径。不要为了增加论文感先加入复杂模块；在首个可用checkpoint和闭环指标出来之前，不扩大本任务范围。

## 14. 参考来源与核对入口

以下是官方原始资料，接口可能变化，实施时记录实际commit/version。架构、超参数和训练课程是本任务提出的工程选择，不是这些来源宣称的现成最优方案。

```text
[S1] DexYCB official reader: split definitions, target index, paths
https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py

[S2] DexYCB official README: data fields and annotations
https://github.com/NVlabs/dex-ycb-toolkit
https://dex-ycb.github.io/

[S3] FoundationPose actual pose storage and return conversion
https://github.com/NVlabs/FoundationPose/blob/main/estimater.py

[S4] Torchvision ResNet50 pretrained-weight API
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

[S5] NVIDIA nvdiffrast documentation and CUDA rasterizer
https://nvlabs.github.io/nvdiffrast/
https://github.com/NVlabs/nvdiffrast

[S6] PyTorch distributed overview
https://docs.pytorch.org/tutorials/beginner/dist_overview.html

[S7] torchrun
https://docs.pytorch.org/docs/stable/elastic/run.html

[S8] Codex CLI official documentation
https://developers.openai.com/codex/non-interactive-mode
https://developers.openai.com/codex/agent-approvals-security
```
