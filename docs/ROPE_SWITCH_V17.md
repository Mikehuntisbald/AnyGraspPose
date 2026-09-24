# V17 frozen RoPE on/off

Evaluate step24400 (`ec0732f609fd04dc7aee50a42c1ae7b9830a9e9acf8f1ac9a614efc45e50add5`) on the existing fixed40 protocol, with natural/light/heavy8/heavy16/heavy32 cases. Eight GPUs shard the same40 physical sequences. No optimizer, training, official test or additional seed.

Both arms share one model instance, one encoded observation and one teacher target per frame. On uses the learned four `cad_surface.rope3d.gain` values; off temporarily sets only those gains to zero. Restore the original values after every forward, including exception paths. History is disabled in both arms; all parameters are frozen. The entire model state digest must match before and after each shard. This measures current-checkpoint inference dependence, not the effect of training a separate model without RoPE.

Keep the same config, CAD cache, crop/base trajectory, fixed step11000 EMA teacher and occluder plan. Verify paired frame identity, raw feature metrics, target mask counts, teacher CAD entropy and constant-center errors. Log latent/geometry/feature/CAD-log-probability changes as well as reconstruction metrics. There are8000 paired frames (16000 output rows). Aggregate frames per sequence/case, then heavy durations within sequence; report on-minus-off and a paired sequence bootstrap interval.

Run `tools/evaluate_recovery_focus.py --config configs/jepa/dpt_rope3d_v17.yaml --checkpoint <terminal.pt> --out <new-directory> --rope-ablation`. Default evaluation behavior remains unchanged without this flag. Source/config snapshots and shard manifests accompany the results; original training runtimes remain sealed.

CPU checks cover selective parameter intervention, restoration after exceptions, rejection of trainable/history-enabled models, and compiled attention responding to in-place gate changes. Full-protocol results are recorded in the experiment report after completion.
