# V72 fast: local CAD projection learns modest gains; geometry goal remains unmet

Candidate completed200updates and all paired probes. Historical V68r1control200 is reused only after exact first2-update replay across8ranks (loss, gradient norm, metrics, LR and windows).35tests passed. Both checkpoints have verified complete-state resume; all200-step rank logs have finite losses/gradient norms and matching window metadata/LRs. This does not independently hash every training pixel. Evaluation uses the same32/64/32physical-holdout frames at0/10/60degrees, with identical region/flow-point counts and no confidence filtering.

| Heavy condition | Canonical XYZ mm: control | Candidate | Depth mm: control | Candidate |
|---|---:|---:|---:|---:|
| 0deg real-hidden |11.071|10.267|10.008|9.901|
| 0deg CAD proxy |9.252|8.559|8.812|8.393|
| 10deg real-hidden |11.255|10.770|9.726|9.691|
| 10deg CAD proxy |11.139|11.370|9.662|9.344|
| 60deg real-hidden |35.424|35.278|18.361|18.160|
| 60deg CAD proxy |37.481|36.822|18.982|18.731|

At10deg/heavy real-hidden,24/31frames improve XYZ, mean improvement0.486mm (4.3%). At0deg/heavy the real/proxy XYZ gains are about7.3%/7.5%. However,10deg/heavy proxy XYZ regresses0.231mm,0deg/nonheavy real XYZ regresses1.088mm, and large-error recovery remains inaccurate. Compared with the pretraining source,10deg/heavy real XYZ is nearly unchanged (10.798→10.770mm) and depth is worse (8.749→9.691mm). Do not substitute the matched-control improvement for a claim of broad absolute accuracy. No default promotion or training-budget extension.

V70/V71established a decoder failure: exact canonical priors could be corrupted by unrestricted appearance retrieval. The new decoder limits selection to8geometric neighbors and propagates final XYZ gradients to the continuous DPT proposal. This fixes the demonstrated readout behavior but does not make the learned proposal accurate. Predicted correspondence quality, sparse/dense spatial representation, and preservation of the input CAD geometry remain unresolved. Next compare the original dense estimated-CAD geometry directly with the compressed/decoded representation before another training expansion.

Performance: the slow trial was stopped at14updates. GEMM shortlist32 plus direct-distance reranking reduces the synthetic H20 search benchmark284.49→5.87ms. All sampled neighbor indices matched; on real first2training updates, losses and metrics matched but gradient norms differed by up to0.002087, so this is not universal bitwise equivalence. Actual fast-run warm-step median is0.65135seconds versus roughly8–9seconds in the stopped trial.

Candidate final checkpoint: remote local_projection_joint_v72_fast/surface/seed42/last.pt,474007204bytes, recomputed SHA2569a5605822cf9a5407193735a7989fe435d8774e48b6e09a3fcb41cb1e1e7cbec. Historical control remains surface_joint_v68_r1/control/seed42/last.pt, SHA25600454f996a8536b7d1f004fb529edf6794e683b32cb2e6a4d0c20a8fcf3505b7. Checkpoints remain remote; compact raw logs, paired records, source receipts and plots are local. Remote project root is a runtime copy, not a Git checkout. No pose training, official-test evaluation or additional seed was run.
