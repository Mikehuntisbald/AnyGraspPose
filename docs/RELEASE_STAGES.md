# Publication stages

The staged series starts at `fa0905147b4d7f581e25f6b075b600b3ea945bfe`. Commits separate the accumulated experiments while preserving the final working source and all existing runtime directories.

| Stage | Commit | Scope |
|---|---|---|
| 1 | `1986c41` | Frozen FP state transitions, deterministic history mixtures, continuous basin targets, critic fitting/gating, and hybrid evaluation |
| 2 | `1761f5b` | V1 observation preloading, ordered parallel decoding, loader and memory tuning |
| 3 | `e2a8fac` | Streaming single/dual trackers, immutable source KV, explicit migration/resume, training and evaluation preflight |
| 4 | `f1e5f6d` | Batched streaming geometry/state preparation and eight-GPU throughput tuning |
| 5 | `1558a62` | Causal object-to-context attention, gated residual, separate context KV, migration from trained dual |
| 6 | `51d72c2` | Evaluation launch/comparison tools, geometry diagnostics, experiment receipts and public navigation |

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

## September 16 accumulated experiment publication

The next series starts at `51d72c2` and preserves the final working implementation and all existing local/runtime artifacts. The first stage groups the mutually dependent model/config/checkpoint/state changes, rather than publishing an intermediate model registry with missing modules. Tool-dependent tests accompany their tools in the second stage.

1. **Core streaming models and training:** residual cross readout, reliability/keyframe and pose-reference experiments, rotation alignment, spatial supervision, real train initializers, deterministic sampling, startup occlusion, renderer reuse, pose validation and recovery boundaries. Includes the core tests and updated benchmark/preflight interfaces.
2. **Evaluation and experiment workflows:** non-GT inference and BOP export, matched FP comparisons, frozen interventions, geometric diagnoses, experiment controllers, factorial statistics, collection and hash verification. Includes the corresponding tool-dependent tests.
3. **Documentation and compact evidence:** protocols and completed results, navigable factorial figures/statistics/receipts, publication test logs, and an ignore policy that keeps new runtime trees external.

Publication validates isolated Git-index snapshots with the existing remote Python environment, using CPU-only `pytest tests -m "not cuda and not foundationpose"`. Tests do not use or modify the sealed training runtime. Exact tree identities, counts and logs are recorded in [publication checks](../reports/publication_20260916/results.json). Previous GPU/real-data preflights remain historical evidence, not GPU tests rerun for this Git publication.

New `runs/` artifacts remain local. Compact completed evidence under `reports/` is copied with provenance hashes; the source archives, raw datasets, checkpoint files and full prediction traces are not added. Historical documents can still reference full-workspace artifacts that are unavailable in a fresh checkout. Pushing this series updates GitHub history, not the isolated server training directories.

Publication checks for the new series:

| Stage | Commit | CPU tests passed |
|---|---|---:|
| Core | `02e00c7` | 179 |
| Evaluation/tools | `159be89` | 284 |

The second suite includes the first; these are not 463 distinct tests. Both committed source trees passed with zero failures. The documentation/evidence stage does not change their Python code.
