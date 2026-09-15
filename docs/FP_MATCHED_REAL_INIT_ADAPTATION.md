# 同条件 FP 对比与真实初值训练（2026-09-15）

**终态更新：两组 1,000 步与全量验证均已完成，真实组总体严格值
84.489%、严重遮挡 32.099%，但启动旋转短板未解决，仍按预设规则
保留旧 residual。见 [完整完成报告](FP_REAL_INIT_RESULTS_20260915.md)。
以下为启动与运行中记录，保留当时的状态表述。**

当前保留的独立 LIP 已在总体严格精度、中心误差和部分遮挡指标上
超过同条件 FP，但**尚未实现全部指标领先**。本轮实际生成了 train
split 的真实 PoseCNN 初值，并启动两组固定 1,000 步重训；新权重的
完整验证结果尚未产生。以下完成结果与运行中工作分开记录。

两组已保存第 250 步 checkpoint；四个 rank 的 RNG、scheduler=250、
sampler=16,000 和模型张量有限性已额外核验，正式训练继续运行。

## 已完成：同初值的完整 FP 基线

- native s0 val：320 条 grasped-object 流、23,200 帧、20 类对象。
- 两种方法使用同一份冻结的 RGB PoseCNN s0 epoch16 首个合法输出。
  已初始化 319 条流；578 个尚无初值的帧保留在成功率分母中。
- 初始帧保留外部位姿；之后各自递推自己的状态，无 GT pose 初始化、
  无重新检测、无 GT reset。LIP 全程零 FP。
- FP 使用原始 `track_one(iteration=2)` 和 frozen refiner，原生 FP16；
  不调用 registration，不额外 refine 初始帧。已验证 original/centered
  mesh 坐标转换。跳过未使用的 scorer 和 registration rotation grid。
- GT pose/visibility 仅在独立 scoring 阶段读取；推理访问守卫记录为零
  GT pose/mask/hand 输入。FP 的代码/权重读取仅在独立 FP 基线显式放行。
- 这是 native val 的 object-macro ADD 阈值指标；visibility 是 native
  scoring proxy。**不是官方 test BOP AR，也不是同范围 SOTA 结论。**

| 指标（越高越好） | FP | 当前 LIP |
|---|---:|---:|
| 总体 ADD < 0.1d | 52.189% | 57.194% |
| 总体 ADD-S < 0.05d | 72.419% | 83.664% |
| visibility < 0.5，ADD < 0.1d | 37.553% | 27.163% |
| visibility < 0.3，ADD-S < 0.05d | 21.389% | 29.278% |
| 坏初值前八帧，ADD < 0.1d | 63.845% | 54.967% |
| 坏初值前八帧，ADD-S < 0.05d | 82.600% | 79.494% |

| LIP − FP | 点差 | 共享物理序列 bootstrap 95% CI |
|---|---:|---:|
| 总体 ADD < 0.1d | +5.005 pp | [+0.713, +9.153] |
| 总体 ADD-S < 0.05d | +11.245 pp | [+6.773, +15.293] |
| visibility < 0.5，ADD < 0.1d | −10.390 pp | [−16.413, +0.303] |
| visibility < 0.3，ADD-S < 0.05d | +7.889 pp | [−0.081, +18.635] |
| 坏初值前八帧，ADD < 0.1d | −8.878 pp | [−12.537, −4.450] |
| 坏初值前八帧，旋转误差 | +2.950° | [+1.329, +4.465] |

区间使用 40 条物理序列、2,000 次共享抽样，未做多重比较校正。

visibility <0.5 子集仍有 20 类，但 object 11 只有两帧：FP 的 ADD
成功率为 100%、LIP 为 0%，单独贡献宏平均差值中的 5 pp。主分母
保持不变，逐类帧数与完整区间同时保留；不据此删除不利样本。

总体中心误差 LIP / FP 为 11.305 / 43.801 mm；坏初值前八帧为
13.984 / 14.479 mm，后者差值区间包含零。由此，本轮优先针对
**启动旋转恢复和遮挡 ADD**，同时保护既有中心优势；不能把前一轮
“新结构对旧 LIP 的中心退化”直接当作“LIP 对 FP 的中心短板”。

恢复分析保留 114 个遮挡事件，77 个具有完整十帧 clear 窗口，37 个
被边界、下一次遮挡等截尾；不把截尾当作恢复失败。稳定恢复需连续
三帧 ADD-S < 0.05d 且 status ok。各方法“遮挡末帧失败”的条件分母
不同，单独描述；配对分析固定使用旧 LIP 定义的失败事件。

目前精度评估表的 65 个预列条目中，LIP 有 55 个点值有利、10 个不利。
这些条目高度相关、不是 65 个独立证据；也未包括尚待隔离测量的延迟
与完整设备显存，不能据此声称“所有指标都赢”。

## 已完成：真实 train 初值与预检

27,775 个独立训练图像请求得到 26,764 个合法 PoseCNN 输出，1,011 个
请求无合法输出。候选按检测分数选择，不按 GT 误差筛除；只读指定的
train 帧，不向后搜索，不访问 val/test 图像生成训练数据。

固定 64,000 个训练片段中，真实初值分支请求 32,124 次，实际可用
30,956 次；1,168 次显式回退到共同 noisy-GT 初值。实际真实初值占
48.37%，不能把它写成 50% 已成功初始化。真实初值不再叠加位姿噪声。

原始 PoseCNN 的独立重跑本身存在输出差异；相关原始输出和失败的
跨运行精确一致性检查保留。检查改为对**同一次真实网络输出**验证
原始与重构的预处理、NMS/位姿转换逐位一致，2 张实际图像通过。
实验以冻结的缓存输出和 SHA 为身份，不承诺每次重新执行检测器得到
相同字节。没有改动其 native Hough CUDA kernel。

实际通过：

- 49 项 CPU 训练/状态/采样/初始化协议测试。
- 4 项 CUDA 特征与 cross 测试。
- 7 项 FP 配对统计及遮挡事件边界测试。
- 独立 FP 基线的 26 项协议测试，以及 3 流、220 帧 GPU 边界检查。
- 真训练 RGB-D + PoseCNN 初值：CPU FP32 8 次更新、GPU FP32/BF16
  各 16 次更新，训练接口与逐帧在线接口的输出逐位相同。
- 真实两片段、三步 FP32 梯度 pilot：loss 0.067778 → 0.063587 →
  0.059804；所有可训练梯度有限、RGB 不变。pilot 权重和 optimizer
  已丢弃，这不是正式训练精度结果。
- 两组 batch16 显存探测、四卡 DDP 3 步及恢复到第 4 步，所有 rank
  的 RNG 与固定采样位置恢复检查通过。单卡 PyTorch 峰值约 30.6 GB。

第一次隔离 runtime 的 pytest 继承父目录配置并导入旧 `src`，导致
9 个 collection errors；没有进入训练。修复为明确绑定隔离项目的
`pyproject.toml`，旧失败日志保留，新 runtime 的测试如上通过。

## 正在执行的固定对照

父权重：已选定的 residual，保持全部初始模型张量逐位相同，fresh
optimizer。两组分别用四张 H20，同时运行，seed42：

| 项目 | control | real_mix |
|---|---|---|
| 起始位姿 | noisy GT | 50% 确定性请求真实 train PoseCNN；缺失显式回退 |
| 初始 RGB-D | 先编码，保留给定位姿 | 相同 |
| 训练预算 | 1,000 步 × 64 片段 | 相同 |
| 每片段 | 初始观察 + 56 次更新，48 个监督目标含前八帧 | 相同 |
| 遮挡配方 | S1O1：概率 0.5，启动遮挡比例 0.5 | 相同遮挡图案与随机种子 |
| RGB 主干 | 全阶段冻结 | 相同 |
| 其余参数 | 小学习率适配，loaded 2e-6、new 类别 1e-5 | 相同 |
| FP / MANO / 手监督 | 无 | 无 |

两组遮挡图案均从同一个训练 noisy prior 生成，避免真实初值改变
遮挡增强位置造成混杂。初始观察的编码、帧计数与 online priming
一致，预测不在初始帧被直接采纳。

正式训练输出：

```text
/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/pair/
  status.json
  control/train/rank0.jsonl ... rank3.jsonl
  real_mix/train/rank0.jsonl ... rank3.jsonl
```

完成后的自动工作已排队：冻结两组 final1000、在当前源码下完整复现
旧 residual 轨迹与阈值分数、各跑完整 320 流 / 23,200 帧，然后做
共享初值/可见性/帧身份的配对筛选。旧权重复现不过或测试失败就停止。
结果目录：`runs/full_val_v2/`。旧 `runs/full_val/` 只是被替换的等待
控制器，没有推理结果；替换原因是统一 benchmark 的 CPU 位姿输出边界。

晋级条件：总体 ADD 与严格 ADD-S、严重遮挡严格分数的点值不得低于
旧 residual；在预列 FP 短板中，至少一个对旧 residual 的配对区间
改善。点值保护不是统计非劣证明。固定只看 final1000，不挑中间步。
未达标保留旧权重，不自动启动新 test。

之后单独运行 FP、旧 residual、被选候选的 batch1 延迟测量：固定
20 类流、每类 32 次更新，CPU RGB-D 输入至 CPU original-mesh 位姿
输出，CUDA 同步；排除 IO/模型加载/首观察 priming，包含预处理、
render、传输和时序更新。PyTorch 峰值与设备显存采样分开记录，
不能将并发全量评测耗时称为独立延迟。

## 身份与证据路径

- 父权重 SHA：`89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`。
- 本轮 `src/lip` SHA：`e054bc82476241a13278e06ac5adf9409b99df36c625930685e9911144e03698`。
- val 初值 SHA：`ea33a4668526d2305f5a23d4e789d579094f0f2639ac81386c93766329e574bf`。
- train 初值 SHA：`3c4bd539702506d8a05d8172783e948bdd73b72dd850a3b301bf109a831ba3bd`。
- PoseCNN checkpoint SHA：`f2cc07f4e61b3077b76175459b5bace6405180e7efb70add115857d595d0a6a6`。
- FP refiner SHA：`774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60`。
- FP source commit：`a1b694b83e633c2cb6115b9063d940a687759392`。

FP refiner 是此前保存的镜像权重；本轮核验的是上述文件 SHA，不能
将其等同于独立核对了不可访问的原始 Drive 文件。

服务器绝对路径：

```text
FP 全量：/mnt/why/dexycb_lip/fp_val_20260915/runs/full_v2/scored/
训练初值：/mnt/why/dexycb_lip/posecnn_train_20260915/runs/full/initializers.json
旧 LIP 对 FP 的配对表：/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/fp_residual_comparison/
本轮预检：/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/pair/
本轮自动验证：/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/full_val_v2/
```

已导出的 [图表与报告](../runs/fp_real_init_20260915_snapshot/fp_residual_figures_v2/report.md)
包含固定 167 条坏初值流的 0–8 帧曲线。可见 FP 的旋转改善集中在
最初几帧；LIP 的严格成功率约在第 5 次更新追平，中心误差随后更低。
这是当前固定总体的描述，不是新训练已经改善的证据。

当前仍待完成：正式 1,000 步训练、final1000 完整验证、独立延迟和
设备显存记录、新候选冻结。未产生这些终态凭据前，不报告本轮改善。

## 当前训练命令与故障恢复

以下是两组实际使用的训练入口（服务器上执行；输出目录已运行中，
不要重复启动）。配置、数据清单、preflight 均为本轮固定文件：

```bash
cd /mnt/why/dexycb_lip/real_init_training_20260915_v2
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /mnt/why/dexycb_lip/.venv-fp/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m lip.train_stream --config runs/pair/control/config.yaml --output runs/pair/control/train \
  --max-steps 1000 --data-root /mnt/why/dexycb_lip/cache/raw_full_20260910 \
  --index-root /mnt/why/dexycb_lip/cache/dexycb_s0 --fixed-manifest runs/pair/training_samples.json \
  --init-from runs/pair/control/init.pt

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /mnt/why/dexycb_lip/.venv-fp/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m lip.train_stream --config runs/pair/real_mix/config.yaml --output runs/pair/real_mix/train \
  --max-steps 1000 --data-root /mnt/why/dexycb_lip/cache/raw_full_20260910 \
  --index-root /mnt/why/dexycb_lip/cache/dexycb_s0 --fixed-manifest runs/pair/training_samples.json \
  --init-from runs/pair/real_mix/init.pt
```

若对应训练进程确实已退出且 `train/last.pt` 已保存，在同目录、同配置、
同四卡与固定清单下，将相应命令的最后一项替换为：

```bash
--resume runs/pair/control/train/last.pt
# 或另一组：
--resume runs/pair/real_mix/train/last.pt
```

恢复上限仍为 1,000 步，恢复 optimizer、scheduler、各 rank RNG 与
sampler；不是再额外训练 1,000 步。失败的控制器还需要核验并重新接续
其终态审计/评估，不能只看到 checkpoint 就声称整个流程完成。
