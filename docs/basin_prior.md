# Basin-aware candidate prior

The critic predicts frozen FoundationPose success for a candidate correction in the current LIP context. Inputs are the 256-dimensional latent z and six correction coordinates: camera-frame rotation vector in radians and mesh-center translation divided by object diameter. Candidate pose construction uses the same decoupled Update as the LIP head. The critic outputs a temperature-calibrated logit.

For each target frame, collect six candidates around the frozen LIP prediction: original; paired +5 degree/+0.02 diameter and -5 degree/-0.02 diameter perturbations; +15 degree rotation; +0.05 diameter translation; and a random rotation up to 30 degrees plus translation up to 0.1 diameter. Axes and magnitudes are seeded and saved. Every candidate independently initializes the same frozen FP refiner and runs two refinement iterations on the same RGB-D frame. Success is ADD-S < 0.1 diameter after FP, computed from GT only after refinement. No candidate's FP state carries into another candidate.

This estimates a local, finite-iteration success region under the generated candidate distribution, not an unrestricted or asymptotic convergence guarantee.

## Data and fitting

Version 1 predeclares 2,048 unique target frames, six candidates each, sampled exclusively from official s0 train. History mixtures use 40% noisy GT, 30% LIP-only, 30% LIP-to-FP; the frozen actor supplies four-step histories, and the final step supplies the context to label. GT teacher histories remain strictly past relative to the target frame. All candidates for each physical sequence remain together in one of train/calibration/test groups (80/10/10 of physical sequences). Official validation and test sequences are never used for critic fitting or gating.

LIP and FP are frozen during collection. The actor checkpoint, refiner weights, source configuration, sequence split and dataset contents are hash-bound to receipts. Each frame stores z, candidate delta, candidate poses, FP-refined poses, labels and metadata. Latent features are never paired with candidates from a different frame or base pose.

Train unweighted BCE with AdamW; select the epoch on calibration-group NLL, then fit a scalar temperature on the same calibration groups. The held-out sequence test is evaluated after fitting. Predeclared gates require at least 30 positive and negative candidates, 30 mixed-label frames, AUROC >= 0.70, within-frame positive/negative pair ordering >= 0.60, and Brier/NLL better than the training-prior constant predictor. These are engineering acceptance criteria, not statistical proof that applying the prior improves LIP.

## Actor integration

Only a critic checkpoint with a passing receipt may be loaded. Freeze critic parameters and keep it in eval mode, but do not put its forward pass in no_grad: gradients must flow through candidate delta. Use z.detach() to block the direct conditioning-feature gradient:

```python
logit = frozen_critic(z.detach(), delta_lip)
L_basin = softplus(-logit).mean()  # stable -log sigmoid
L_total = L_pose + lambda_basin * L_basin
```

The delta path still trains the pose head and upstream LIP features. This does not make z immutable across actor updates; feature-distribution drift must be checked before activation from a newer actor checkpoint. FP remains a no-grad environment transition, distinct from this differentiable critic prior. The existing 40/30/30 -> 20/20/60 history curriculum is retained.

Default proposed prior weight is a 1,000-step ramp from 0 to 0.01, gated by critic validation and real gradient/DDP preflight. It must remain disabled if critic acceptance fails. Report whether the critic was merely trained or actually accepted and enabled; do not infer actor benefit from critic AUC alone.

Files: `configs/basin_critic.yaml`, `src/lip/models/basin.py`, `tools/collect_basin.py`, `tools/train_basin_critic.py`, and `runs/basin_v1/` for real receipts. Production training uses isolated candidate source directories; local edits do not replace active training code.

## Continuous outcomes (schema 2)

The enriched archive `runs/basin_v1/outcomes_v2/outcomes.npz` stores arrays with leading dimensions `[frame, candidate]` and a `frame_files` mapping to the immutable source records. It preserves `z`, `delta`, GT and split metadata along with:

- `candidate_pose`, `refined_pose`: mesh-centered object-to-camera transforms, with `*_pose_original` aliases for original-mesh coordinates.
- `add_before_over_d`, `add_after_over_d`, `adds_before_over_d`, `adds_after_over_d`.
- `rotation_before_deg`, `rotation_after_deg`, `center_before_mm`, `center_after_mm`.
- Signed `delta_add_over_d`, `delta_adds_over_d`, `delta_rotation_deg`, `delta_center_mm`: before minus after.
- `e_before`, `e_after`, `delta_e`, `y_converge`, `y_improve`.
- `diameter_m`, `mesh_center_m`, `target_metric`, `target_tau`, `target_margin`, `targets_schema_version`.

User-confirmed policy: e = ADD-S/d; `y_converge = (e_after < 0.1)`; `y_improve = (e_after < e_before - 0.005)`. Comparisons are strict. Negative delta_e is retained for regressions. The legacy `labels` field aliases y_converge only, so older convergence-only loaders cannot silently switch targets. Two future probability heads must consume their respective target arrays; continuous after-error and signed gain remain available for regression/ranking even when both binary labels agree.

The 2,048-frame / 12,288-candidate schema-2 archive was reconstructed from saved candidate/GT poses and actual frozen-FP outputs. No FP rerun was needed. Original files and the failed first critic receipt remain unchanged. The enriched archive does not by itself qualify the first critic for use in the actor loss.

```bash
PYTHONPATH=src python tools/enrich_basin_outcomes.py \
  --source runs/basin_v1/data --out runs/basin_v1/outcomes_v2 \
  --metric adds_over_d --tau 0.1 --margin 0.005
PYTHONPATH=src python tools/verify_basin_outcomes.py \
  --data runs/basin_v1/outcomes_v2 --samples 32
```

The schema helper `lip.data.basin_targets` is also used by new collection runs, so newly sampled boundary candidates preserve the same continuous outcomes and labels at creation time.

## Validated boundary critic and main-training integration

The user explicitly chose to improve the critic and require acceptance before integration. The v1 critic and v2 dual-head critic both failed candidate-order gates and were not used in production. Their receipts remain intact. No experimental bypass is enabled in the validated continuation.

The v3 data adds one far probe (30–90 degree rotation and 0.1–0.3 diameter center translation) and, if endpoint convergence labels differ, three bisection probes on that ray. Every label is from actual frozen FP. Bracketing is a sampling procedure and does not assert a globally monotone convergence basin. The original six candidates are preserved. Variable candidate counts are masked in all training losses and metrics. V1 test frames are excluded; v3 uses 320 fresh validation frames from 70 physical sequences absent from all v1/v2 critic data.

V3 contains 2,152 frames and 18,772 candidates. The critic predicts calibrated convergence/improvement logits plus log post-FP error, trained with BCE, continuous-error regression and within-frame ordering. A frozen copy of the source actor head maps z to a reference correction, so the candidate encoder can use both absolute and relative delta. Inference still requires only z and candidate delta. The convergence logit is monotone in predicted post-FP error at fixed z.

Fresh-group results: convergence AUROC 0.9186, improvement AUROC 0.7829, and continuous post-FP ordering 0.8861 over 300 frames / 6,060 candidate pairs. Both heads improve on the training-prior constant predictor in Brier and NLL. On the saved step-20,000 actor's latent, all criteria still pass (ordering 0.8836). These are critic checks inside s0 train, not evidence that the main actor has improved.

Validated actor loss remains `L_pose + lambda * (-log q_converge(z.detach(), delta))`. The improvement head is an auxiliary critic target and is not used to reward intentionally poor initial poses. Both FP and critic weights stay frozen; only delta's path through LIP receives the extra gradient. Lambda ramps from 0 at step 20,000 to 0.01 at step 21,000. The FP-aware history mixture and all main pose losses remain in place.

The earlier FP-aware process exited with SIGKILL after logged step 20,285. NVSwitch/GPU Xid records occurred in the same time window; causality is not established. All eight GPUs subsequently reported no recovery action and passed NCCL / control-group health checks. Recovery uses the immutable step-20,000 checkpoint; the unsaved 285 steps are replayed. Optimizer, scheduler and sampling state are restored. No host driver or CUDA changes were made.

Runtime source/config: `runs/basin_boundary_v3/candidate/`. Evidence: `fit/receipt.json`, `drift.json`, `gradient_probe.json`, `gpu_health.json`, `ddp_probe.log`, and `preflight_passed.json` within `runs/basin_boundary_v3/`. Main log: `runs/basin_boundary_v3/train.log`. Main rank rows include `run_id=basin_boundary_v3` to distinguish replayed steps from the earlier segment.

Recovery after an interruption (do not duplicate an active training job):
```bash
cd /mnt/why/dexycb_lip
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_full_20260910
bash scripts/train_basin_8gpu.sh --resume runs/lip_v1_s0/last.pt
```
The first continuation uses `runs/basin_boundary_v3/resume_20000.pt`; subsequent saved checkpoints embed the basin configuration and external critic hash.
