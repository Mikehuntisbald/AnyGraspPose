# V54 full validation of JEPA recovery

The user requested full evaluation after V53 completed. This evaluates V52 balanced200 versus V53 final5200 on all320 s0-validation camera streams belonging to40 physical sequences, all23200 indexed frames. Natural input,25% light and80% heavy additional RGB-D occlusion are evaluated, yielding69600 paired conditions. Synthetic coverage is calibrated to the estimated CAD silhouette; actual retained visibility is also recorded.

## Fixed protocol

Both models receive the same crop, estimated-pose CAD rendering, real RGB-D, geometry, state and occluder draw. Source/candidate raw inputs are compared bitwise for every paired forward. The input base is the previous frame from the sealed uncorrupted LIP trajectory, or the non-GT PoseCNN initializer on its first valid frame. Current-frame synthetic occlusion is drawn independently per case/frame; this is not an occluded closed-loop tracking or temporal-memory experiment. History stays off.

GT masks and current-pose teacher renders are built only after both student predictions. One shared target build supplies both models. Original reconstruction masks are preserved; no extra training quarantine or prediction-confidence filtering changes score denominators. Artificially hidden originally-visible regions retain sensor depth; naturally hidden proxy regions use CAD depth; unmasked visible preservation is scored separately using sensor depth.

- CAD/object-frame XYZ error measures surface identity correspondence, without symmetry reduction.
- Camera XYZ is recovered depth lifted through calibrated pixel rays; depth and surface-normal errors evaluate physical geometry.
- GT-relative projection of predicted CAD identity provides an image-space correspondence error for scoring only.
- Validity Brier/TPR/FPR and all-target valid XYZ<10mm/depth<5mm fractions expose incorrect confident completion.
- Full coverage is reported:22622 post-initialization frames are eligible to form inputs;578 earlier/uninitialized frames remain as unavailable records. Missing baseline poses and crops with no GT surface are counted explicitly, not silently removed. Error means still require legal input and nonempty region targets.

Aggregation: frames within each camera stream, equal camera streams within physical sequence, equal physical sequences. Report cases and target sources separately; include per-object, natural-visibility, retained-visibility and base-rotation strata. Framewise medians/p90 values are averaged hierarchically; they are not pooled pixel percentiles. The unequal-frame/camera-count sanity case gives12.5, confirming the declared weighting rather than frame pooling.

## Execution and provenance

V52 SHA256 `a6bd6204e1c646e31a02c25fd20d9cf7051354fb0e1e04b7d2f083111fc2df81`; V53 SHA256 `717b503de54724e7dad79fe667b9fd49fe392b898af2522267add59ee65bdb3d`. Weights are frozen; no training, pose solver, default-model change or official-test access.

Active pinned source: `/tmp/dexycb_full_recovery_v54_r1`. Active artifacts: `/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_fullval_v54_r1`. Seven target/metric tests and8×2-frame paired smoke tests passed before full evaluation. The first preflight was correctly rejected for duplicate lane stream identities; each case now has its own stream identity. The failed snapshot/logs remain preserved separately at `recovery_fullval_v54`; no full results came from that attempt.

The previous128 sampled development frames used a controlled10-degree GT-perturbed base in the training split. Absolute errors from this full native-reference validation must not be ranked against those short probes without acknowledging the population/input differences. Full evaluation remains conditional on the common baseline crop and does not establish autonomous pose performance.

## Completed result

All69600 paired conditions are accounted, with22622 legally initialized frames and578 unavailable frames per condition;215 initialized crops contain no GT surface. Re-reducing every downloaded row reproduces the remote aggregates exactly, and all8 shard hashes match. Heavy real/proxy CAD identity XYZ29.717/31.464→28.996/31.018mm; depth12.623/14.850→11.717/14.033mm. Correspondence projection errors remain16.324/18.270px. This does not meet accurate-recovery goals. Natural visibility<20% shows essentially no native proxy-depth gain. Full results and limitations: [summary](../reports/jepa_20260921/unified_rgbd_v2/recovery_fullval_v54/full/SUMMARY.md).
