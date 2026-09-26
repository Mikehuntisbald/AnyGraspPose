# V35: isolate learning of the CAD correspondence decoder

V34's new path contributed only about2.4% and its flow was worse than identity.
This100-update diagnostic changes only the transport head; all pre-existing
JEPA, DPT, encoder, EMA and pose tensors stay fixed. Parent is V34 transport100,
SHA256`eb8f8bbf729b0b6cf2d71ab18a536e88ede3a5611dc9dd38603fea2deeecbd00`.

The head transfers V34 weights, except the declared gate bias reset to0
(sigmoid0.5). Its own step0 is evaluated: gains caused by changing the initial
mixing weight must not be credited to training. AdamW resets, LR1e-3,
10-step warmup, cosine floor0.5, seed42,8H20,batch32. Exactly100 updates,
checkpoint every50, explicit2→100 strict resume. New training seeds35000000
remain outside the physical holdout. No DINO/pose loss or history is enabled.

The same64 training-partition physical-holdout records are evaluated before
and after, including zero-flow and current-estimate CAD overlap diagnostics.
Successful learning requires lower flow EPE than identity and improvement
over the changed-gate initial checkpoint, together with real/proxy XYZ/depth
improvement. A more heavily used CAD prior alone is not sufficient.

Runtime`/tmp/dexycb_geometry_transport_warmup_v35`, artifacts
`/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_transport_warmup_v35`.
No automatic budget extension or default-model promotion follows this probe.

## Completed: no continuation

Equal-physical-sequence heavy real XYZ/depth:13.708/10.497mm at changed-gate
initialization,13.796/10.289mm after100updates. Proxy15.283/14.191mm becomes
15.996/14.870mm. Flow5.897/5.009px is worse than identity5.219/4.463px.
Both acceptance gates fail. All699 non-transport tensors and the complete
2-step resume are verified exact; only the decoder optimizer is populated.

Full terminal checkpoint and all94 delivery files are SHA-verified locally
under reports/jepa_20260921/unified_rgbd_v2/geometry_transport_warmup_v35.
The next V36 comparison adds reference conditioning from the SAME initial
checkpoint and reuses the same100-update data/schedule; it does not continue
V35's unsuccessful terminal weights.
