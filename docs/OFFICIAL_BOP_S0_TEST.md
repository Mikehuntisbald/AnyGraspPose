# Frozen residual 1,000 on official DexYCB s0 test

This is a single-image pose-refinement benchmark, not the earlier GT-initialized tracking protocol. The selected checkpoint is fixed to SHA256 `89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`; no training or test-driven hyperparameter selection is performed.

## Inputs and inference

The initializer is the published PoseCNN RGB prediction CSV from the official DexYCB CVPR 2021 results archive. The downloaded archive matches the publisher's MD5 `6b1e883d2c134ffe6228fe727d90650a`. No GT pose or GT mask is used for initialization. The public BOP target object IDs and instance counts are the known-object localization inputs.

For each official target image independently:

1. Select the requested number of PoseCNN candidates using their original confidence scores.
2. Produce PoseCNN-only, PoseCNN → LIP, PoseCNN → FP(2 iterations), and PoseCNN → LIP → FP(2 iterations) predictions from the same candidates.
3. LIP uses one update, an empty source/context cache, and synthetic `dt=1/30` to encode a single-image refinement state. No previous test frame is used. FP's state is reset to the supplied candidate before each call.
4. Preserve the input pose if a refinement returns an invalid pose, logging the event. Unexpected software failures abort the worker. Missing initial detections remain missing for all variants and remain in the official scoring denominator.

A runtime audit hook rejects inference-side Python opens of raw pose labels, scene GT, masks or MANO annotation/calibration paths, and rejects writes to the raw dataset. RGB/depth image paths are constructed explicitly. Scoring is a separate process that is allowed to read GT. The official dataset constructor reads hand calibration metadata while constructing its dataset, but no hand model or hand annotation enters the predictor or its supervision.

Population: **1,280 scenes, 23,644 target keyframe images, 88,014 object targets**. The common initializer supplies **85,929 estimates**; **2,085 targets lack an initializer**. Full `all` and `grasp_only` official aggregates are reported separately.

## Coordinate and evaluation contracts

LIP and FP operate on the original DexYCB textured mesh with meter translations. The input CSV for the DexYCB official wrapper stores original-mesh poses with translation in millimeters and rotation flattened row-major. Official `BOPEvaluator._convert_pose_to_bop` then applies the published per-object origin shift exactly once. Converted standard BOP model-coordinate CSVs are retained alongside the input CSVs.

The official toolkit is pinned at `64551b001d360ad83bc383157a559ec248fb9100`, its BOP submodule at `035da77330823779c796272ca125c43a0b7e5ffa`. The metric source is unmodified. `tools/evaluate_official_bop.py` invokes the official `eval_bop19.py` on disjoint scene target lists, concatenates the official match records, and calls the official result aggregation. It does not average per-shard AR.

VSD, MSSD and MSPD error functions, threshold grids, visibility selection, symmetry handling and target counts come from the pinned official evaluator. A private `.venv-bop` plus compatibility launcher restores removed Python/NumPy aliases and registers the existing ctypes pointer handler for modern Python. Official glumpy rendering runs under private Xvfb displays using Mesa llvmpipe. No host driver, global CUDA, or raw data changes are made. Compatibility failures and their logs are retained; successful results must come from the final working evaluator run.

## Interpretation

The three refinement pipelines use identical PoseCNN initial candidates, RGB-D observations, camera intrinsics and known meshes. PoseCNN alone is an RGB initializer reference. Released DeepIM RGB-D predictions provide an additional official 2021 reference; they are not labelled current SOTA. Training data/budgets differ, notably DexYCB-trained LIP versus the released FoundationPose refiner. This single-image application does not measure the benefits of LIP's temporal history.

Runtime: `/mnt/why/dexycb_lip/official_bop_residual_1000`.

- Frozen protocol and inference logs: `runs/official_s0_test/`
- Official PoseCNN reproduction: `runs/official_posecnn_final/`
- Complete paired CSVs: `runs/official_s0_test/csv/`
- Final comparison/report: `runs/official_s0_test/comparison.json` and `report.md`
- Live phase/progress: `runs/official_s0_test/status.json`

`tools/finish_official_bop.py` waits for complete inference manifests, verifies identical candidate populations, and schedules official scoring and the final report. A `completed` status is required before interpreting scores; this document does not itself assert that the long evaluation has finished.

Sources: [official toolkit](https://github.com/NVlabs/dex-ycb-toolkit), [official results download](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/results/fetch_cvpr2021_results.sh), [official evaluator](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/bop_eval.py).

## Completed run

The full evaluation completed successfully. Official PoseCNN reproduces all AR 52.682% and grasp-only AR 41.648%. PoseCNN → LIP → FP reaches all AR 72.835% and grasp-only AR 64.843%, versus 71.408% / 61.309% for PoseCNN → FP with the same initial candidates. Every method retains all 88,014 valid GT targets. See [full report](../runs/official_s0_test/report.md) and [completion receipt](../runs/official_s0_test/completion_verified.json). These are official s0-test refinement-pipeline results, not a claim to lead a current SOTA leaderboard.
