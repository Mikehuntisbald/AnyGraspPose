# Matched FoundationPose baseline

Completed all 320 official s0 validation camera streams and 23,200 frames. GT initialization only on each first frame, followed by closed-loop tracking. LIP checkpoint: step 10,000.

| Metric (object macro) | LIP | Pure FoundationPose |
|---|---:|---:|
| ADD-S@0.1d (%) | 95.42 | 85.98 |
| ADD@0.1d (%) | 76.83 | 69.05 |
| Center error (mm) | 11.33 | 40.56 |
| Rotation error (deg) | 13.80 | 20.01 |
| Lost-frame rate (%) | 3.36 | 11.81 |

ADD-S@0.1d difference: +9.44 percentage points for LIP. This is validation performance of a DexYCB-trained model versus frozen FoundationPose weights; not a final test or equal-training-data comparison.

Measured mean serial latency: LIP 28.52 ms/frame, FoundationPose 39.98 ms/frame. Both synchronize CUDA and include raw decoding; exclude GT metrics and mesh setup. Their native precision/preprocessing differ (LIP BF16; official FP FP16 AMP), so this is the measured implementation performance, not an identical-kernel benchmark.

The exact mirror weight revision and SHA-256 values are in weights_receipt.json; official Google Drive files were quota-limited. Protocol and implementation details: ../../docs/foundationpose_baseline.md.

Training was paused at checkpoint 13,000 with zero replayed steps and resumed automatically; all eight ranks were observed past 13,000 with finite losses, matching scheduler and sampler positions. The new live log is training_resumed.log on the server.
