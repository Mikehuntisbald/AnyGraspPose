# V74: dense CAD preservation works; large-error alignment remains unsolved

Both arms completed200updates from the same V72-fast checkpoint, seed42,8H20s. Training preflight:38tests passed. Full-state resume and all-rank finite gradients passed. Dense-label projection was independently checked with float64 native intrinsics plus crop transforms. All4new-head parameter tensors changed from initialization, ruling out accidental freezing. Actual final checkpoint hashes were recomputed.

Evaluation uses32/64/32pose-conditioned cases at0/10/60degrees inside the training partition. These are controlled development cases, not128independent sequences or native deployment/official-test results. Original target masks and point counts are preserved across arms. Heavy means requested synthetic augmentation. The128-case supplemental audit exactly reproduced primary geometry and sparse-flow metrics.

| Heavy condition | Canonical XYZ mm: control | Candidate | Depth mm: control | Candidate |
|---|---:|---:|---:|---:|
| 0deg real-hidden |8.747|0.600|8.041|8.024|
| 0deg CAD proxy |12.425|1.195|8.702|8.960|
| 10deg real-hidden |10.007|9.412|8.633|8.611|
| 10deg CAD proxy |12.152|9.061|7.968|7.981|
| 60deg real-hidden |34.258|39.791|16.057|16.180|
| 60deg CAD proxy |38.838|39.157|15.064|14.492|

The zero-error arm intentionally supplies a correct estimated reference; its strong gain proves preservation of useful input geometry, not general pose-error recovery. At10deg/heavy, proxy XYZ improves25.4%, real-hidden5.9%. At60deg, real-hidden worsens16.2%; nonheavy real-hidden also regresses44.234→65.369mm. Depth has no consistent improvement. No default promotion or achieved-accuracy claim.

The new dense backward-flow audit explains the limit. At10deg/heavy real-hidden, learned EPE5.0509px versus zero-flow5.0544px; proxy4.3107versus4.2447px. At60deg/heavy real-hidden,26.7867versus26.8943px. Thus strong alignment learning is not established; results are consistent with gains dominated by preserving the reference. The gate is on82.9%of60deg/heavy real-hidden pixels because its training label describes correspondence existence (GT support86.1%), not whether the actual read is better than fallback.

Next-stage code adds an opt-in teacher-only read-preference target comparing detached actual-read and fallback canonical errors, with0.005diameter margin. Flow and sampled-XYZ supervision remain active on fixed supported targets even when the predicted gate closes. Fortyregression tests pass under the correct deterministic-CUDA environment. This option is NOT trained and was NOT used for V74; availability labels remain the default. Relative branch preference is not absolute confidence, and gating alone will not solve large-error correspondence.

Source commit48448e6matches executed files (executed_source_verification.json). Candidate checkpoint remote surface/seed42/last.pt SHA256122e10bfb36af77cc925df38e180d3aefeffe06d6636e4d2e6c145212c32a622; control SHA2566bb8ec32dc57fbc18e409a11b86ebf99cb9ad73e5b6709cb72c28db4d4ca65cc. Both are under unified_jepa_20260921/dense_canonical_v74. Full checkpoints remain remote; compact raw records, source receipts and plots are local. No pose training, extra seed, official test or automatic default change occurred.
