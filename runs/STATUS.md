# 实施状态：DexYCB-LIP v1

工程主路径已实现，并通过**真实训练子集的工程预检**。完整 s0 数据预检未通过，正式长训练没有启动，当前 checkpoint 不是可用跟踪模型。

本地工程：`/home/haoyi/Downloads/xd/dexycb_lip`。远端工程：`/mnt/why/dexycb_lip`，通过 `ssh -p 10863 root@111.230.4.68` 访问。下列相对日志路径在两处都有副本；checkpoint 与原始子集只保留在远端。

## 实现与真实验证

| 项目 | 实际结果 | 证据 |
|---|---|---|
| 固定架构 | ResNet50 layer3 + 9-channel geometry CNN + 2 spatial cross-attention + 4 causal temporal layers + 256 latent + 解耦 pose correction | `src/lip/models/tracker.py` |
| CPU/GPU 单元测试 | 21 passed, 1 skipped；skip 为缺依赖/权重的真实 FP 推理 | `runs/pytest_final.log` |
| 无手监督、GT 隔离、闭环状态所有权 | 物体索引、多对象非首位置、无 hand 字段前后向、未来帧扰动、当前/未来 GT 改动、padding、短历史、无效 depth、近零/近π梯度、非零 mesh center、checkpoint/RNG、lost 帧保留等测试通过 | `tests/test_core.py` |
| 官方 split | s0/s1 规则已实现；合成完整序列集合验证 s0 为 800/40/160 且物理序列互斥；完整真实 s0 清单尚不可生成 | `src/lip/data/index.py`, `runs/full_data_audit/audit.json` |
| 真实子集 | 官方已知 s0 train 序列 `20200709_141754`，8 个相机流，每流72帧，共576帧；1112个合法 L=8/U=4 候选窗口 | `cache/s0_verified_subset/audit.json`, `cache/raw_train_subset/subset_provenance.json` |
| 几何审计 | 32/32 样本有足够可见像素；每样本可见区域深度绝对误差中位数再取中位数为 4.385 mm；最大 crop 投影误差 0.00006866 px | `runs/subset_geometry/geometry.json`, `*_overlay.jpg`, `*_depth.png` |
| 真实32-clips过拟合 | 按训练混合采样固定32个不重复clips、关闭噪声/增强/dropout，500 optimizer steps。loss 0.08472012 → 0.01560852；ADD 9.577 → 1.589 mm；center 9.618 → 1.486 mm | `runs/overfit32_mixed_subset/report.json`, `manifest.json`, `loss.jsonl` |
| 单卡真实 batch probe | 4/8/16/32 均完成单步与U=4前后向及optimizer step。选32；U=4 peak allocated 18.348 GiB，reserved 20.213 GiB，低于实测初始可用显存80% | `configs/resolved_8gpu.probe.json`, `runs/probe_real_subset.log` |
| 8卡真实DDP | 每卡32、accum=1、有效batch256，50个optimizer steps，全部U=4。8个rank均无非有限值；每rank最大allocated峰值的最大值 18.406 GiB | `runs/ddp_real_subset/rank0.jsonl` 至 `rank7.jsonl`, `runs/ddp_real_subset_launcher.log` |
| 恢复训练 | 从step50恢复到53；8个rank的global step、scheduler step、采样位置一致。未声称GPU逐位确定性 | `runs/ddp_real_subset_resume.log`, `runs/subset_preflight_receipt.json` |
| 真实评估入口 | 单步32样本的三种方法，以及首帧GT初始化后的8流/576帧闭环全部执行；无GT重置，跟丢帧完整保留 | `runs/eval_mixed_subset_onestep/`, `runs/eval_mixed_subset_closed/` |
| 安全的长训练门槛 | 完整s0数据缺失时正式入口拒绝运行；一键preflight亦在数据审计阶段退出 | `runs/formal_guard_attempt.json`, `runs/full_preflight_attempt.json` |

`configs/resolved_8gpu.yaml` 已由真实子集 probe 与8卡短跑验证，明确记录 `engineering_preflight_passed: true`、`official_s0_complete: false`、`preflight_approved: false`。这不是完整s0训练许可。`scripts/train_8gpu.sh` 会检查完整s0/geometry/真实overfit/DDP/resume凭据。

早期单独的 moving-only 32-clips 诊断保留在 `runs/overfit32_subset/`，对应评估在 `runs/eval_subset_*`；上述表格与正式子集验收使用重新固定的**混合采样**结果 `overfit32_mixed_subset`。合成8卡50+3步也保留在 `runs/ddp_synthetic/`，没有替代真实数据短跑。首次系统 torchrun shebang 绕过虚拟环境导致缺包的失败日志也保留；已改用虚拟环境的 `python -m torch.distributed.run` 并实际重跑通过。

## 模型质量仍未通过

上述过拟合与DDP检查验证实现可训练，不证明长期跟踪质量。采用混合采样过拟合checkpoint，在同一个训练序列的576帧闭环中：

- 平均 center error：332.164 mm。
- 平均 rotation error：20.280 deg。
- ADD recall@0.1d：16.493%；ADD-S recall@0.1d：19.965%。
- lost fraction：74.479%（连续5帧ADD-S>0.1d触发；所有帧仍在分母）。
- 同步测得batch=1端到端均值 23.357 ms，p95 24.670 ms；含磁盘读取/解码、crop/render/network，排除GT指标与可视化，不含FP。此延迟不能弥补跟踪漂移。

统一32个noisy-history单步诊断的平均ADD：zero-motion 9.214 mm、constant-velocity 9.953 mm、learned 8.979 mm。这是训练子集诊断，不是held-out泛化证据，亦不能证明latent学会交互。

## 数据与环境边界

- 之前下载的完整119 GiB官方包尚未形成已解压的完整数据根目录。此次没有自动另启整个DexYCB下载，没有修改原压缩包或宿主机驱动/CUDA/PyTorch。
- 子集从已完整下载的官方archive前缀只读提取到项目 `cache/raw_train_subset`；只使用官方README明确列出的train位置0的序列。没有把不完整目录排序冒充完整s0。
- 标定与目标CAD来自HF第三方补充副本，经HF LFS SHA256验证，再通过真实深度/投影审计。它们不是拥有官方发布者checksum证明的全量镜像。来源详见 `cache/assets/receipt.json` 和 `docs/CONTRACTS.md`。
- 本子集只有1个物体/1段真实录制，没有heavy-occlusion桶；采样将moving/uniform明确重归一化为2/3与1/3。多物体、重遮挡、完整val/test、s1重训尚未实际验证。
- 环境为8张H20，每张97871 MiB；PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、nvdiffrast 0.4.0。实际依赖、driver、cgroup资源记录在 `runs/environment.json`、`docs/requirements-remote.txt`。最后检查8卡均为0 MiB使用，训练进程均已结束。
- 没有真实FoundationPose实例/权重，只有独立adapter及非零center/rejection单元测试。没有`best.pt`，因为没有完整val选择；现有`last.pt`是过拟合或短跑checkpoint。

## 后续命令

数据完整下载并解压到正确目录后，在远端前台运行：

```bash
cd /mnt/why/dexycb_lip
source scripts/env.sh
export DEX_YCB_DIR=/mnt/why/DexYCB   # 必须包含完整subjects、calibration、models
bash scripts/preflight.sh          # 完整s0审计和短跑；不会启动长训练
bash scripts/train_8gpu.sh         # 正式40,000-step训练，仅在完整预检通过后可运行
bash scripts/train_8gpu.sh --resume runs/lip_v1_s0/last.pt
```

现有真实子集checkpoint：

- `/mnt/why/dexycb_lip/runs/overfit32_mixed_subset/last.pt`：混合采样32-clips过拟合500步。
- `/mnt/why/dexycb_lip/runs/ddp_real_subset/last.pt`：8卡U=4短跑与恢复后的step53。

复现当前子集训练恢复（最多100步诊断，独立于正式训练入口）：

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train \
  --config configs/resolved_8gpu.yaml --data-root cache/raw_train_subset \
  --index-root cache/s0_verified_subset --allow-verified-subset \
  --output runs/ddp_real_subset --resume runs/ddp_real_subset/last.pt --max-steps 55 --force-rollout 4
```

完整val/test与单步/闭环命令见 `README.md`。此次没有自动启动长训练。
