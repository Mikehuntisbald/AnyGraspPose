# V58: distinguish recovered content from actual measurement evidence

V57 found weak point visibility discrimination (AUROC0.624 on its10-degree
probe) and no point crossing0.5. It also found that soft inclusion of measured
depth could hurt CAD-proxy geometry. This stage tests whether explicit observed
information can fix evidence selection before changing geometric decoding.

Every V56 backbone, DPT, encoder, CAD and flow parameter stays frozen at source
SHA256 `044a8f928d324e54c8e2cab2927dd31e55942cde98bc09fa5f18ce5245d0fa5f`.
The new head cannot alter endpoint predictions, CAD transport writes or ordinary
geometry outputs. It has two outputs: point visibility, and measurement usability
at the **predicted** endpoint. No additional pose or RGB-D encoder is introduced.

## Inputs and ownership

The272-dimensional feature separates139 existing/recovered channels from133
observed-evidence channels. The latter include the matched original DINO
descriptor, its difference from the reconstructed descriptor, actual measured
depth/validity and disagreement with recovered depth. Sampling uses only the
predicted endpoint. All inputs are detached in this deliberately isolated stage.

The restored-only control zeros the133 observed channels. Both arms have the same
MLP, exact same initial weights, minibatches, objective and optimization schedule.
The new code path is opt-in; tests verify that changing its logits does not change
flow endpoints, support logits or the reconstruction write.

Visibility uses the existing audited template-point label. Measurement usability
requires a known front surface, a currently visible real pixel at predicted UV,
endpoint error<=3px, finite positive measured depth and a<=30mm gap to GT CAD
depth. These are explicit quality labels, not a claim of millimeter measurement
accuracy. Unknown boundary labels are excluded. **Original real-depth and
CAD-proxy reconstruction targets remain unchanged.** No label enters inference.

## Fixed experiment

Eight GPUs collect2048 training frames and128 physically disjoint held-out frames
from the training partition. Training includes25% transported empirical training
initializations; other frames use the existing Gaussian perturbation. Holdout
contains10/60-degree controlled references. Features remain FP32 and are bound to
the frozen source hash. No official test or validation partition is used.

Both heads train1000 fixed updates, seed42, AdamW1e-3, batch1024 points, unweighted
binary cross entropy separately normalized per known label. No checkpoint is
selected by holdout performance. Two-update interruption/resume verifies weights,
Adam, RNG and cache/source/software contracts exactly. The head-stage checkpoint
is resumable; full-model exports are explicitly **inference-only**, with original
V56 step200 and an additional evidence_updates=1000 field. They do not pretend to
contain a restored full-backbone optimizer.

Full forwards then compare baseline/control/observed arms on64 new10-degree and32
new60-degree controlled geometry cases. The diagnostic metric fit uses predicted
measurement quality when available. It still retains recovery depth and robust
fitting; no prediction confidence filters the evaluation targets. Main geometry
must remain exactly unchanged in this isolated stage. Better classification alone
does not satisfy the geometry goal, and no default model is promoted automatically.
