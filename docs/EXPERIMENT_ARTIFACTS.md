# Experiment artifacts

Git contains implementation, configs, tests, analysis/launch tools, and the available compact experiment records: metrics, manifests, migration and preflight receipts, test logs/XML, per-stream comparison CSVs, and reports. The streaming summary plot is included explicitly.

Raw datasets, model/optimizer checkpoints, image overlays, full prediction/training JSONL files, tensorboard event files, runtime source copies, and temporary worktree patches are excluded from new commits. Original tracked bring-up records remain in history. Some report links target these external files and therefore work only in the complete experiment workspace, not in a fresh GitHub checkout.

Checkpoints and full per-frame outputs remain in the experiment server's `runs/` directories. No checkpoint was uploaded as part of this code publication. A published receipt binds the recorded file hashes and configuration; it does not provide a missing checkpoint or certify a changed deployment.

Historical records are preserved as generated, including failed checks and claims that code was then uncommitted or training had not yet started. These are timestamped observations. Publication does not rewrite their historical Git IDs or claim that a later run still has the same status. README tables cite completed validation records, not training metrics.

The local repository and GitHub history are distinct from the server's isolated runtime directories, which are not Git checkouts. Publishing commits does not update those directories or restart training. The verified streaming source snapshots used by the staged commits are:

| Snapshot | Source SHA256 |
|---|---|
| Base single/dual | `3d9cd99903161b0ee7b17cc50df80afe5204cb62c0a3674961f87fd161d9d47e` |
| Batched training | `5ccca80a97c2a50a00819dc3d8d6f86c953eae1e8c7a799e45f85330340708e2` |
| Cross-attention | `7fc30d6c2ff7781ab45cfb5339e9dff631338455a4f57b838d636ec234551095` |

The digest follows `lip.engine.stream_checkpoint.source_hash`: sorted Python paths relative to `src/lip`, followed by each file's exact bytes. The commit identities and publication checks are listed in [release stages](RELEASE_STAGES.md).
