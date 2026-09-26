# V53 formal JEPA recovery training

User requested formal training after the V52 weighting comparison. Continue the V52 balanced200 complete checkpoint for5000 additional updates, global steps200→5200. This does not promote a default model.

## State and learning rate

Source: `/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_balance_v52/balanced/runs/seed42/last.pt`, SHA256 `a6bd6204e1c646e31a02c25fd20d9cf7051354fb0e1e04b7d2f083111fc2df81`.

`geometry_horizon_continuation` overrides the inherited V52 initialization descriptions and declares the active checkpoint. Model/EMA, all Adam moments and step counts, scheduler state, all8-rank RNG and sampler position are restored exactly. The scheduler function is explicitly extended: starts at saved0.5×peak LR, reheats continuously for100 updates to the same peak, then cosine decays to0.5×peak at5200. Optimizer is not reset. Subsequent process restarts use strict resume under the V53 config/provenance.

Peak LRs: DINO1e-6, JEPA3e-5, DPT1e-4, CAD atlas1e-3. The first step uses the saved half-peak rates. Existing losses, architecture, data, augmentations, gradient clipping and optimizer settings remain unchanged; the extension validator rejects unrelated changes.

## Scope and monitoring

- 8 H20s, seed42, effective32 observations/update, two estimated-pose hypotheses per observation.
- Train DINO/JEPA/DPT/CAD interaction; pose modules frozen; history, pose loss and DINO feature loss off. Correspondence CE weight0.1.
- Real sensor targets remain real; naturally hidden CAD targets remain proxy; boundaries/conflict masks and original evaluation masks remain unchanged.
- Every50 global steps atomically saves a full checkpoint. Global700/1200/2700/5200 (new500/1000/2500/5000) preserve immutable checkpoints and sequentially evaluate both fixed64-frame recovery sets.
- Report canonical CAD XYZ, physical camera XYZ/depth and camera-surface orientation separately. These are development holdouts already inspected during V52, not a new unseen-test claim.
- No pose/PnP validation, default-model replacement, automatic hyperparameter changes, multi-seed or official test. Stops on runtime/nonfinite failures or the fixed budget boundary; metric changes are reported.

Runtime: `/tmp/dexycb_recovery_formal_v53`. Artifacts/status/live logs: `/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_formal_v53`. Existing `.venv` is reused unchanged. Startup includes29 passing tests and a two-update extension, followed by strict202→700 resume. Source archives/hash receipts remain immutable.

The V52 candidate is a useful continuation point, not a complete repair: the ordinary heavy probe has a proxy correspondence/depth tradeoff. Monitor both sets and regions throughout formal training.

## Completion verified on2026-09-27

All5000 new updates and all scheduled recovery evaluations completed. Global terminal step5200, full checkpoint SHA256 `717b503de54724e7dad79fe667b9fd49fe392b898af2522267add59ee65bdb3d`; all four strict resumes verified. Training is no longer running. [Final monitoring](../reports/jepa_20260921/unified_rgbd_v2/recovery_formal_v53/REPORT.md). The ordinary heavy-depth errors improve16.76%/12.86%; confirmation heavy-depth errors improve only1.31%/1.95%, and confirmation all-frame real depth worsens2.24%. Canonical correspondence improves on both sets. Accurate recovery remains incomplete; no automatic promotion or extension. This progress refresh synchronized metrics/receipts, not the terminal weight file.
