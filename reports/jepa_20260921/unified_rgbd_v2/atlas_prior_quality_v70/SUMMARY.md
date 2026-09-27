# V70: separate inaccurate priors from decoder-induced geometry error

64frozen V60-extra cases, exact V69seeds; normal real/proxy geometry records match V69. All depth and flow values remain unchanged. Primary evaluation retains every original target pixel; covered-only fields are diagnostic, not improved-denominator metrics.

Two distinct problems are measured:

1. Predicted flow does not produce accurate dense canonical priors. At10deg/heavy real-hidden, transported prior error15.381mm versus original DPT12.991mm; at60deg it is39.766mm versus36.741mm. Around99%coverage does not imply correct correspondence. Even oracle-supported sparse anchors have about11mm covered-region prior error, showing the nearest-anchor spatial approximation is coarse.
2. Final atlas selection can corrupt accurate geometry. With exact canonical XYZ supplied as a diagnostic prior,60deg/heavy real-hidden output error is23.025mm and proxy19.466mm. Tightening sigma from0.1to0.03reduces these to5.119/4.494mm. This is causal evidence of decoder-induced displacement, not proof that the learned prior is accurate.

Blindly tightening predicted priors is not a fix:60deg/heavy predicted-flow real-hidden final error rises37.896→38.993mm, proxy38.755→41.849mm. The evidence motivates a geometry-preserving local CAD decoder plus improvement of the continuous proposal, rather than a global sigma change.

All slices, prior/covered/uncovered errors and coverage are in REPORT.md/outcome.json. Oracle variants are diagnostic only. No training or model promotion in V70.
