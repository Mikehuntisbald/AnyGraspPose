# Independent LIP occlusion evaluation

The primary question is LIP's own causal tracking robustness, without requiring
FoundationPose. The network accepts RGB-D and its own previous pose/history. An
external initial pose is still necessary because LIP is a refiner/tracker.

Primary run: `runs/lip_only_s0_test` under remote runtime
`/mnt/why/dexycb_lip/stream_bop_residual_1000_20260913`.

- Frozen residual 1,000 checkpoint, unchanged SHA256
  `89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868`.
- Direct first legal PoseCNN prediction as initial pose; no GT pose input.
- Zero FP calls, including initialization. No FP modules or weights are loaded.
  The inference audit hook rejects access to the FP installation path.
- Two branches: full causal LIP, and LIP with feature history cleared while
  retaining accepted poses/motion. Each recursively commits its own predictions.
- Every intermediate RGB-D frame is processed. No redetection or GT reset.
- Same s0 test official target population: 88,014 target instances, with 87,043
  expected predictions and 971 targets missing a causal initialization.
- Report official AR and posthoc BOP visibility strata for all/grasped objects,
  plus paired physical-sequence bootstrap intervals for the history effect.

`standalone_preflight.json` verifies a real 32-object-frame smoke run, zero FP
imports/calls, cache growth to eight frames, and initialization CSVs exactly
matching the direct PoseCNN pose after float32 conversion. The 32 inference
workers run four per H20. The full inference, official scoring, occlusion
aggregation, and local artifact collection are now completed and hash-verified.

## Completed zero-FP primary result

Full LIP obtains all-object AR **72.2770%** and grasped-object AR **70.0599%**.
The grasped-object occlusion breakdown is:

| Visibility | No feature history AR | Full LIP AR | Gain (pp), 95% paired interval |
|---|---:|---:|---|
| >=0.5 | 71.4768 | 72.3727 | +0.8959 [0.6974, 1.0739] |
| <0.5 | 50.5917 | 52.4094 | +1.8177 [1.1978, 2.5543] |
| <0.3 | 39.5591 | 41.7529 | +2.1938 [0.8496, 3.6343] |

Feature history improves occluded-subset accuracy under zero-FP inference, but
the absolute performance still drops substantially under heavy occlusion.
The counts are 20,132 / 2,638 / 970 target instances, respectively; the last two
subsets overlap. No valid official targets have visibility <0.1.
See `runs/lip_only_s0_test/conclusion_zh.md`, `occlusion_ar.json`, and
`local_collection_verified.json` for the conclusion, paired evidence and hashes.

## Secondary initialization condition

The earlier `runs/streaming_s0_test/lip_temporal` result is a secondary initial-pose
condition: its tracking phase uses only LIP, but initialization uses PoseCNN→FP.
Its measured BOP-visibility AR is:

| Visibility | All objects AR (%) | Grasped objects AR (%) |
|---|---:|---:|
| >=0.5 | 77.3622 | 74.0995 |
| <0.5 | 49.4496 | 53.2739 |
| <0.3 | 38.6835 | 42.2134 |

These numbers describe LIP after the shared FP-refined initial pose. They must
not be presented as full-pipeline zero-FP results or as an isolated history gain
against earlier single-image scores. The <0.3 subset is nested inside <0.5;
there are no <0.1 instances in the official valid-target population. BOP visibility
is distinct from the Visibility Aware paper's hand-projection visibility metric.

The completed paired history ablation under that SAME shared initialization
isolates the effect of keeping feature history in the subsequent LIP-only loop:

| Grasped-object visibility | No feature history AR | Full LIP AR | Gain (pp), 95% paired interval |
|---|---:|---:|---|
| >=0.5 | 73.1892 | 74.0995 | +0.9102 [0.7236, 1.0889] |
| <0.5 | 51.4064 | 53.2739 | +1.8676 [1.3719, 2.4922] |
| <0.3 | 39.9069 | 42.2134 | +2.3065 [1.5219, 3.2240] |

Intervals use 1,000 paired physical-sequence bootstrap resamples. These are
descriptive subset results and establish a positive history effect in this run;
they do not isolate the newly added cross-attention branch or eliminate the
initialization-condition limitation. Exact metrics and cluster-level inputs are
in `runs/streaming_s0_test/occlusion_ar.json`.
