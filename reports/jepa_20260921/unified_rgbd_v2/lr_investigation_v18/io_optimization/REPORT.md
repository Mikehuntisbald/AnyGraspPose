# Local execution with durable training artifacts

Resumed low-LR replay from complete step26100, SHA256 `e039f0a7e8fd5ee2f6854a53efc26cd64b6dd8a30ad30065bfc23aa35e888956`. Strict training resume validated model, Adam, scheduler, all-rank RNG, sampler, configuration and provenance. Model/training source remains byte-identical. Only controller supports separate execution/artifact roots and skipping completed training targets.

516 pinned files copied and checked. Execute from `/tmp/dexycb_lr_investigation_v18_local`; preserve logs, checkpoints, evaluations and status in original Lustre artifact directory. CAD source-hash validation remains intact; its repeated reads now hit local disk. No cache contract relaxation, new CAD cache identity, model/loss/LR change or reset.

| Measurement | Before:26001–26100 (100 steps) | After:26120–26149 (30 steps) |
|---|---:|---:|
| Mean seconds/update | 6.084 | 3.221 |
| Median seconds/update | 5.658 | 3.052 |
| Teacher seconds/update | 1.664 | 0.397 |
| Gradient communication seconds/update | 0.442 | 0.277 |

Observed speedup1.89x, step time reduced47%. Different adjacent episodes; this is runtime throughput evidence, not a paired microbenchmark. First compile/warmup excluded. Sampling400 main-thread states after warmup found331 running /69 futex waits and no Lustre wait states, versus prior observed Lustre waits. This does not guarantee future absence of shared storage stalls; datasets and durable writes still use existing storage paths.

Continuation controller still completes replay26400, fixed40 LR selection, then resumes preserved main26500 to35400. Source/config controller commit5cfb6e9 pushed to origin/main. Remote primary checkout Git HEAD not asserted; isolated runtime hashes are the synchronization evidence. `/tmp` is ephemeral; durable source and checkpoints remain on Lustre and can recreate the execution copy after reboot.
