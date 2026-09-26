# Recovery-only V52 delivery

Both200-update arms and the prespecified confirmation64 completed. No GPU job is pending. No default weights were replaced and no pose experiment was launched.

- Local full checkpoints: `control/runs/seed42/last.pt`, `balanced/runs/seed42/last.pt`. Each contains full training state. Their hashes match the remote terminal records in `checkpoint_verified.json` and per-arm `local_checkpoints_verified.json`.
- Remote original artifacts: `/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_balance_v52`.
- Pinned execution source: `/tmp/dexycb_recovery_balance_v52`, preserved without in-place changes. Per-arm `source.tar.gz` archives and file-hash receipts are retained locally and remotely.
- Code and pre-training audit commit: `87cded0`. Later reporting/plotting and verification code is delivered in a separate commit.
- Remote project root is a runtime file copy, not an aligned Git checkout. Synchronizing source/doc/report files does not claim remote Git HEAD alignment.

26 targeted tests passed per arm.715 initial model tensors match exactly; all8 ranks have identical initial-forward metrics.35 pose tensors remain bitwise unchanged. Strict2→200 optimizer/scheduler/RNG continuation passed. The64 confirmation frames have zero exact stream/frame overlap with the usual64; both derive from the same training-partition physical holdout pool. No official test or multiple seeds.

Accurate recovery remains unfinished: depth errors are around10mm, and the usual probe's proxy correspondence/depth trade off against real recovery. Keep both checkpoints and both probes visible; do not claim the lower-weight model wins every region.
