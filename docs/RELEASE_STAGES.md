# Publication stages

The staged series starts at `fa0905147b4d7f581e25f6b075b600b3ea945bfe`. Commits separate the accumulated experiments while preserving the final working source and all existing runtime directories.

| Stage | Commit | Scope |
|---|---|---|
| 1 | `1986c41` | Frozen FP state transitions, deterministic history mixtures, continuous basin targets, critic fitting/gating, and hybrid evaluation |
| 2 | `1761f5b` | V1 observation preloading, ordered parallel decoding, loader and memory tuning |
| 3 | `e2a8fac` | Streaming single/dual trackers, immutable source KV, explicit migration/resume, training and evaluation preflight |
| 4 | `f1e5f6d` | Batched streaming geometry/state preparation and eight-GPU throughput tuning |
| 5 | `1558a62` | Causal object-to-context attention, gated residual, separate context KV, migration from trained dual |
| 6 | this documentation/evidence commit | Evaluation launch/comparison tools, geometry diagnostics, experiment receipts and public navigation |

Stages 3 and 4 are assembled from the retained, previously tested source snapshots; stage 5 matches the current cross-attention source. Git's index was used to form these commits without replacing the user's working files or active server source.

Each of the first five committed trees is exported into an isolated directory and checked with the existing project environment using `pytest tests -m "not cuda and not foundationpose"`. These checks use CPU only and do not interrupt training. Their actual outcomes and logs are published under `runs/release_stages_20260912/`.

The prior cross-attention preflight separately passed 100 distinct CPU/CUDA tests, real 16-fragment / 300-step overfit, bounded 10,000-update cache checks, and eight-GPU 50-step plus 3-step resume checks. Those historical GPU checks are recorded in `runs/stream_v2_cross_8800_preflight/`; they are not described as rerun during publication.

Data, checkpoints and full JSONL traces remain external as described in [artifact policy](EXPERIMENT_ARTIFACTS.md). No force push, history reset, driver change, or runtime synchronization is part of this publication.

Publication CPU checks completed successfully (zero failures in every stage):

| Commit | Passed tests |
|---|---:|
| `1986c41` | 44 |
| `1761f5b` | 45 |
| `e2a8fac` | 72 |
| `f1e5f6d` | 76 |
| `1558a62` | 92 |

See [machine-readable results](../runs/release_stages_20260912/results.json) and the adjacent per-commit logs/XML.
