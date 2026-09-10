# 上传后自动训练

用户已于2026-09-10明确授权“完成后开始训练”。远端 tmux 会话为 `dexycb_full_train_20260910`。

任务按顺序等待13个分包的完整上传回执，核对文件名/大小/两端SHA256及gzip校验标记，再重新核验压缩包并解压到项目独立的 `cache/raw_full_20260910`。接着运行完整官方s0索引、32样本几何审计、8卡采样统计、单元测试、真实32-clips/500步过拟合、单卡batch probe、8卡50步U=4短跑及恢复到53步。全部通过才调用原来的前台 `scripts/train_8gpu.sh` 启动40,000 optimizer steps；如果该正式输出已有匹配checkpoint，则恢复它。任何失败都停止并写入 `status.json` 和对应阶段日志，不跳过验收。

已有的单序列与三位受试者预检结果保持独立。新的完整数据索引为 `cache/dexycb_s0`，训练输出为 `runs/lip_v1_s0`，数据环境变量由任务设置为 `DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910`。

当前状态必须读取远端实时文件，本地副本只是采集时的快照：

```bash
ssh -p 10863 root@111.230.4.68 'cat /mnt/why/dexycb_lip/runs/full_train_after_upload_20260910/status.json'
ssh -p 10863 root@111.230.4.68 'tail -f /mnt/why/dexycb_lip/runs/full_train_after_upload_20260910/workflow.log'
```

正式训练开始后，详细日志位于 `runs/full_train_after_upload_20260910/train_40000.log`，逐rank训练记录位于 `runs/lip_v1_s0/rank*.jsonl`。任务最后会核验checkpoint和8个rank均达到step40000，之后才生成训练完成回执。训练开始前的 `training_started=false` 指正式训练尚未启动，不代表没有执行短预检。

启动前另外修复了完整验证等待的通信方式：长时间rank-0验证使用独立Gloo CPU组，训练仍使用NCCL。实测8卡在NCCL超时设为10秒、rank-0验证模拟12秒后仍成功执行all-reduce。证据为 `validation_wait_test.json`；本次准备检查为 `setup_tests.log`（28 passed / 1 optional FP skipped）及 `autostart_final_tests.log`（4 passed）。

原始脚本依然支持前台执行；本次用户授权的等待和接续工作由远端tmux管理，不依赖持续的SSH连接。上传完成前，本地上传服务仍需电脑和网络在线。

## 监控期间的准备优化

已校验分包提前解压，按文件加锁，因此与完整接续流程并行也不会重复写同一目标。最后一个subject-07镜像仅在整包SHA256与用户本地文件完全相同时采用，原始分块保留。

完整上传回执仍须包含13个包。训练准备解压其余12个包（10位受试者、calibration、models）；`bop.tar.gz`是另一套评估格式，主模型reader不依赖它，保持单独后台解压，不阻塞完整原始RGB-D的s0预检。`verified_upload.json`保留全13包，`training_archives.json`明确记录训练依赖范围。
