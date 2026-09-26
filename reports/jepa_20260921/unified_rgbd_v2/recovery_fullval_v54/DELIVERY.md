# Full recovery validation delivery

Completed V52 balanced200 versus V53 final5200; all320 validation streams/23200 frames,69600 paired natural/light/heavy conditions.

- `full/SUMMARY.md`: interpretation and limitations; `full/REPORT.md`: generated metric definitions and paired tables.
- `full/outcome.json`: sequence-balanced metrics and all object/visibility/base-rotation strata.
- `full/rank*/frames.jsonl`: all paired per-frame records. Shard hashes are bound by manifests; local recomputation exactly matches remote aggregates.
- `full_frame_artifacts.tar.gz`: archived raw full-evaluation outputs; `full_frame_artifacts_verified.json` binds its SHA256. Large raw records/archive stay as workspace/runtime artifacts; compact results/manifests/plots are versioned in Git.
- `full/*.png`: summary, base-rotation strata and predetermined error-map examples.
- V53 complete terminal checkpoint is locally verified at `../recovery_formal_v53/runs/seed42/last.pt`; SHA256 `717b503de54724e7dad79fe667b9fd49fe392b898af2522267add59ee65bdb3d`.

Original remote results: `/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_fullval_v54_r1`. Pinned runtime: `/tmp/dexycb_full_recovery_v54_r1`, unchanged after launch. The earlier duplicate-lane preflight failure remains preserved separately; full evaluation used the corrected pinned snapshot. No default weights were replaced or extra training started.

The remote project root is a file-synchronized runtime copy, not a Git checkout; synchronization does not assert remote Git HEAD alignment.
