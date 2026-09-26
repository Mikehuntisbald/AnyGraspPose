# JEPA recovery gradient balance V52

Only correspondence weight changes: control1.0, balanced0.1. Network architecture and all learning rates are unchanged. Encoder/JEPA/DPT/CAD train, pose/history/DINO loss stay off. Matched data and strict resume checked.

The lower weight improves real-depth recovery on both probes. The usual probe trades worse proxy correspondence/depth for better real recovery; the prespecified confirmation improves both regions. This is evidence that weighting contributes to the problem, not a complete geometry repair or a uniform win. No default-model promotion.

## paired: heavy subset

|Arm|Real CAD XYZ mm|Real depth mm|Proxy CAD XYZ mm|Proxy depth mm|Real camera XYZ mm|Proxy camera XYZ mm|
|---|---:|---:|---:|---:|---:|---:|
|control_0|14.422|12.505|18.336|13.360|13.029|14.031|
|control_200|13.298|12.558|15.300|12.796|13.088|13.424|
|balanced_0|14.422|12.505|18.336|13.360|13.029|14.031|
|balanced_200|12.278|10.424|16.566|13.026|10.858|13.700|

## confirmation: heavy subset

|Arm|Real CAD XYZ mm|Real depth mm|Proxy CAD XYZ mm|Proxy depth mm|Real camera XYZ mm|Proxy camera XYZ mm|
|---|---:|---:|---:|---:|---:|---:|
|source|13.781|12.731|12.347|9.978|13.102|10.262|
|control|12.769|13.283|13.050|11.353|13.663|11.657|
|balanced|11.604|11.859|11.725|9.653|12.199|9.902|

The usual64 probe is reused development evidence. The confirmation64 uses seeds54000000+rank+8*draw, fixed before training and sampled from the same held-out physical-sequence pool. It checks new frames/occluders, not unseen objects or a new sequence split. No candidate is automatically promoted.

[Paired recovery scores](paired_recovery.png) · [Predetermined heavy-frame depth/error maps](fixed_recovery_example.png) · [Checkpoint verification](checkpoint_verified.json)

26 targeted tests passed in each arm. Checkpoint verification records715 identical initial tensors and35 unchanged pose tensors; first-forward metrics match on all8 ranks. Full2→200 resume is verified. Real depth targets remain sensor measurements; proxy targets remain GT CAD; original evaluation masks remain unchanged.
