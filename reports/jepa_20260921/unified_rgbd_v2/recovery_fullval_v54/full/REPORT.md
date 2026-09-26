# V54 full validation: V52 versus V53 recovery

All320 camera streams /40 physical sequences /23200 frames accounted; natural/light25%/heavy80% input conditions,69600 paired conditions. Synthetic fractions refer to the estimated CAD silhouette; actual retained visibility is reported.

The student uses the same previous sealed LIP pose/crop (PoseCNN at initialization) for both weights. GT supplies scoring only. This is full-population conditional reconstruction, not closed-loop tracking, pose evaluation, or a controlled10-degree initialization experiment.

|Condition|Evaluated native frames|Unavailable initialization|Missing reference|No GT surface in evaluated crop|
|---|---:|---:|---:|---:|
|natural|22622|578|0|215|
|light|22622|578|0|215|
|heavy|22622|578|0|215|

## Missing-region recovery

|Condition/target|CAD identity XYZ mm V52→V53|Depth mm V52→V53|Camera XYZ mm V52→V53|Camera normal deg V52→V53|
|---|---:|---:|---:|---:|
|natural/real|— → —|— → —|— → —|— → —|
|natural/proxy|30.714 → 30.392|13.375 → 12.490|13.792 → 12.864|26.576 → 23.405|
|light/real|27.816 → 26.890|11.253 → 10.224|11.601 → 10.543|26.482 → 24.321|
|light/proxy|30.864 → 30.531|13.673 → 12.856|14.098 → 13.242|26.711 → 23.695|
|heavy/real|29.717 → 28.996|12.623 → 11.717|13.028 → 12.094|27.796 → 25.763|
|heavy/proxy|31.464 → 31.018|14.850 → 14.033|15.307 → 14.454|27.369 → 24.499|

Real means originally visible then synthetically hidden: original sensor depth. Proxy means naturally hidden: rendered CAD depth. Unmasked visible regions are scored separately against sensor depth. No predicted-confidence masks filter the errors. Empty target regions are counted, not assigned zero error. Means are conditional on legal initialization and nonempty targets; coverage above is part of the result.

## Actual correspondences and valid recovery

|Condition/target|Identity projection error px V52→V53|All-target valid XYZ<10mm AND depth<5mm fraction V52→V53|
|---|---:|---:|
|natural/real|— → —|— → —|
|natural/proxy|18.122 → 17.942|0.097 → 0.113|
|light/real|15.376 → 14.963|0.174 → 0.216|
|light/proxy|18.229 → 18.024|0.092 → 0.106|
|heavy/real|16.657 → 16.324|0.128 → 0.153|
|heavy/proxy|18.516 → 18.270|0.080 → 0.091|

Per-object, original-visibility, retained-visibility and base-rotation strata, visible-region preservation and validity calibration are in `outcome.json`. Rotation bins use unreduced GT-relative base angles for scoring; their populations differ and cannot establish causality. No default model was promoted.
