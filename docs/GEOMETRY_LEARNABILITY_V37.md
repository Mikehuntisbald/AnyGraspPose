# V37: can the correspondence decoder fit a small, fixed target?

V34–V36 did not beat identity flow. This diagnostic distinguishes loss
interference from limited decoder/readout learnability before another long run.
It does not train or replace the production JEPA model.

Source: V36 initial checkpoint, SHA256
`6bf4c1b03e68c8814373b2ce6834574cd3476e7053cf552df7759d4923edd6cc`.
All encoders/JEPA/DPT/pose modules are frozen. Eight workers each cache one
training and one physical-holdout scene from the TRAINING partition. Each scene
uses the SAME RGB-D crop/occluder with positive and negative10-degree estimated
rotation around a declared camera axis. GT targets remain separate; student
inputs contain only the noisy estimate and its reference render. Original RGB-D
and teacher canonical targets are verified identical within each pair.

Minimum eligibility is128 missing-geometry pixels and128 reference-supported
correspondence pixels. Filtering uses target availability, never model error.
The cache stores final DPT inputs, detached predicted fallback geometry,
estimated-reference geometry and final JEPA patches. Labels are never decoder
inputs. Hashmod5 physical holdout remains disjoint from training.

Four isolated diagnostic heads receive identical initial weights:

- Flow-only loss on one record,400 full-batch updates.
- Flow-only loss on eight records (four paired scenes),400 updates.
- Existing mixed geometry objective on one record,400 updates.
- Existing mixed geometry objective on eight records,400 updates.

AdamW1e-3,seed42,gradient clipping1.0,FP32 weights/BF16 convolution match the
short decoder stages. Flow-only uses exactly the prior0.25-weighted normalized
SmoothL1 correspondence term. Mixed uses the existing geometry/normal/gate/flow
objective; coarse, validity and visibility terms have no decoder gradient when
their outputs are frozen. Record train/holdout EPE, identity EPE and geometry
every100 updates. Also measure the initial gradient cosine and norms between
direct flow supervision and all other terms on the SAME head parameters.

Training error can establish learnability only for these examples; it cannot
prove deployable accuracy or generalization. Holdout and whole-region geometry
must be reported even when flow improves. A failed400-step probe does not prove
the full JEPA representation lacks information.

Source is pinned at`/tmp/dexycb_geometry_learnability_v37_r1`; artifacts under
`/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_learnability_v37`.
The first cache launch exposed a NumPy scalar conversion error before any
cache/optimization completed; its logs remain under`_failed_scalar`.

## Completed result

|Diagnostic|Train flow EPE px,0→400|Holdout EPE px,0→400|
|---|---:|---:|
|Flow-only,1 record|3.751→0.183|5.309→6.179|
|Mixed,1 record|3.751→1.126|5.309→6.709|
|Flow-only,8 records|4.127→1.611|5.309→14.732|
|Mixed,8 records|4.127→3.579|5.309→13.460|

The single mixed-loss record reaches real XYZ2.313mm and depth1.663mm,
from8.011/3.044mm. It has NO eligible proxy region, so this is not evidence of
accurate CAD-proxy reconstruction. The model can fit individual geometry and
correspondence examples; the numerical path is not inherently unable to learn.
Generalization is poor and no model is promoted.

For the eight-record batch, other-objective gradient norm3.268 versus direct
flow0.0642 is50.9x; cosine+0.192. For one record cosine+0.919. This shows scale
imbalance and different optimization outcomes, NOT a proven universal negative
gradient conflict. Mixed loss reduces eight-record geometry error but leaves
flow worse than flow-only, consistent with free residuals/gating compensating
for inaccurate correspondence. The next matched trial makes surface identity
explicit, retains raw real-depth targets and tests both label contracts under
the same stronger flow weight and mandatory CAD lookup.
