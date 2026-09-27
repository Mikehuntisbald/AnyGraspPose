# V69: stronger decoder coupling, insufficient predicted correspondences

Frozen V60-extra,64paired frames at10/60degrees. No training, pose solver, or default promotion. Predicted canonical CAD identities are located using forward-flow endpoints, then supplied as a14px-radius canonical prior to the existing atlas search. The final learned atlas scores still select a point from the complete CAD bank. This is a hard-assignment diagnostic, not yet a trainable JEPA redesign.

| Heavy region | Original XYZ mm | Predicted flow prior | Oracle endpoints | Oracle endpoints + supported anchors |
|---|---:|---:|---:|---:|
| 10deg real-hidden |13.4582|12.8905|11.4316|11.3523|
| 10deg CAD proxy |15.1400|15.7839|14.1653|20.6638|
| 60deg real-hidden |37.5776|37.8957|33.1447|28.9090|
| 60deg CAD proxy |37.4540|38.7549|36.3402|31.7312|

Predicted-flow coupling provides a modest10deg real-hidden improvement but harms proxy and60deg geometry. Oracle endpoint interventions now produce measurable geometry changes, unlike the weak latent-write intervention in V61. However, sparse anchor coverage, erroneous predicted correspondences, and interaction with the atlas's learned scores remain limitations. These individual causes have not yet been isolated. The oracle-supported10deg proxy regression is retained; GT-supported-only anchors do not uniformly help.

Depth and both flow rounds were bitwise unchanged, and original region pixel counts were retained for every variant. No covered-pixel-only metric is used. The64normal geometry records exactly match V66. Two coordinate/fallback tests passed. No claim of depth improvement or achieved accurate recovery is made.

Next evidence needed: measure the canonical-prior error and coverage separately from the final atlas selection, and compare accurate-prior lookup with the unchanged learned query scores. This distinguishes sparse/incorrect input evidence from the decoder overriding useful geometry before another training run.
