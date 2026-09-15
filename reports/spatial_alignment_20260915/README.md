# Spatial supervision × direct parent latent

Four fixed-budget arms, each trained for 1,000 steps on the same 64,000 clip draws, then evaluated on native DexYCB s0 val: 320 streams / 23,200 frames. Inference uses real PoseCNN initialization, causal RGB-D and zero FoundationPose calls. The separate frozen FP baseline shares the initializer and population.

See the [interpreted results](../../docs/SPATIAL_ALIGNMENT_FACTORIAL_RESULTS_20260915.md) and [training/evaluation protocol](../../docs/SPATIAL_ALIGNMENT_FACTORIAL_20260915.md). The retained parent was not replaced: auxiliary matching improves on its training targets, but the experiment does not establish improved startup recovery.

- `analysis.json`: all metrics, conditional effects, interaction and bootstrap definitions.
- `contrasts.csv`: paired 95% and 99% intervals; the primary five-contrast family is documented in the report.
- `sequence_object_sums.csv`: cluster/object sufficient statistics for the frame metrics.
- `occlusion_events.csv`: shared event boundaries, censoring and recovery outcomes.
- `training_windows.csv`: 25-step summaries; auxiliary training metrics are not validation accuracy.
- `figures/`: exported plots, in PNG and PDF.
- `verification.json`, `archive_verification.json`, `parent_reproduction.json`: actual experiment and delivery checks.
- `provenance.json`: hashes and original workspace paths for these copied artifacts.

These are one-seed, object-macro native-val results, not official test BOP AR or proof of SOTA. Full per-frame traces, raw data and checkpoints remain external. The archived checkpoint identities and full-archive SHA-256 are preserved in the receipts. Source copies retain their original paths and provenance, even where those paths are not portable to a fresh checkout.
