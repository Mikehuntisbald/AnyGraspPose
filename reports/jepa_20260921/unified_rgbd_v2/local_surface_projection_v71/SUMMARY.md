# V71: locally constrained CAD projection candidate

The candidate restricts final selection to the8geometrically nearest CAD points, with learned appearance selecting only inside that set. The output is an actual CAD point. A straight-through identity gradient additionally connects final XYZ supervision to the continuous DPT prior; this is a biased gradient estimator, not an exact derivative of hard neighbor selection. Default remains off. Seven tests passed, including hostile distant appearance evidence, selected-index consistency, exact missing-CAD fallback and direct coordinate gradients.

Frozen evaluation on64cases, with unchanged depth/flow and original target counts:

| Heavy region | Original | Local projection using predicted DPT prior | Exact-prior diagnostic |
|---|---:|---:|---:|
| 10deg real-hidden XYZ mm |13.458|12.720|2.301|
| 10deg CAD proxy XYZ mm |15.140|18.073|2.583|
| 60deg real-hidden XYZ mm |37.578|36.968|2.569|
| 60deg CAD proxy XYZ mm |37.454|36.096|2.549|

The exact-prior60deg real-hidden error is reduced from V70's original atlas23.025mm to2.569mm. This shows improved preservation of a correct proposal, NOT learned recovery accuracy. The10deg proxy regression prevents immediate promotion. Predicted-flow priors also remain unreliable; V71does not justify forcing them into the decoder.

V72trains this decoder with the existing DPT proposal for200updates, retaining corrected valid-key supervision. It introduces no pose/history training, no GT forward input and no DINO feature loss. The V68r1control's first2updates are replayed across8ranks before reusing its completed200step result; only the candidate receives a new200update run. The goal remains accurate learned geometry, not merely passing the oracle test.
