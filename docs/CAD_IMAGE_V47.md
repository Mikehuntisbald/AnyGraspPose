# V47: frozen JEPA, anchored image displacement readout

Status:500-update frozen-readout trial and fixed40 evaluation completed. It failed to improve recovery/pose; no promotion. User then redirected all further work to JEPA recovery accuracy first.

V45 added explicit CAD-to-image endpoints using transposed descriptor similarity. V46 repaired support/visibility collapse (heavy visible recall0.94%→79.1%), but only15.6% of genuinely visible endpoints are within3px and the40-sequence conditional native test shows no stable pose gain. Geometry also regressed versus the V45 warmup. The V46 checkpoint is not promoted.

This trial isolates readout difficulty. Every V46 encoder, JEPA, DPT, CAD descriptor, visibility and old pose tensor is frozen; EMA updates and running-stat updates are disabled. Only a small new MLP is trained. Its inputs for each of512 fixed CAD points are the DPT query sampled at its **estimated-pose projection**, the existing CAD key and21 geometry channels, pooled DPT context, and the decoded XYZ/depth/validity sampled at that same location. These features are detached by design for this frozen-readout experiment. No teacher enters this path.

The MLP returns a bounded2D residual (56px per axis):

```
CAD point X + base pose -> reference pixel u0
frozen JEPA / DPT information sampled at u0 -> residual du
explicit predicted real-image position u = u0 + du
u + X + predicted reliability -> PnP or RGB-D rigid-fit diagnostic
```

The zero-initialized last layer exactly reproduces the projected base point, so the correct comparison is against this zero-flow starting point, not against V46's inaccurate global argmax. Targets use the previously audited point projections and observed/artificial-hidden/natural-hidden masks. Loss is normalized SmoothL1 endpoint error (weights1/1/0.5, factor40); no other loss trains parameters in this trial. GT-pose examples require zero displacement. Paired estimates share identical observed RGB-D.

This borrows GoTrack's template-anchored flow formulation, while using the existing shared JEPA/DPT and full CAD point identities rather than adding its independent two-branch network. It is not an official GoTrack reproduction. The old learned object-query/pose interface remains preserved; candidate geometric readouts are measured separately before adoption.

Parent: V46 step500, SHA256 `d234ce834c2b1ff280c27c11b8e13a17570c8c5a1d14e6baa7acf3358a8378df`. Runtime `/tmp/dexycb_cad_image_v47`; config `configs/jepa/cad_image_v47.yaml`. Seed42/eight H20s,500 updates, fresh optimizer for the new MLP only, strict2→500 resume,34 targeted tests. Every prior tensor must compare bitwise equal before and after training.

The RGB-D readout tests measured-only, recovered-only and mixed3D points. Mixed mode uses current depth when visibility and a30mm agreement guard pass; recovered total weight is capped by measured total weight when any measurements exist. This is a prediction-only reliability heuristic, not proof of object ownership. Rejected fits retain the base and count in the final metric. Actual CAD correspondences, depth, acceptance and pose quality must all be reported.


## Completed result and priority change

Actual terminal tensors verify all719 previous parameters/buffers unchanged; only32,758 new flow parameters trained. Strict resume passed. On the independent heavy64 probe, endpoint EPE worsened (observed5.107→6.005px, artificial-hidden4.923→5.179px). Fixed40 conditional heavy ADD-S@0.05d fell67.083%→14.583% for anchored PnP; mixed RGB-D stayed50.417%, below the67.083% base. This does not establish that every JEPA representation lacks useful information; this particular frozen readout failed.

V48 began a matched visible-only flow comparison but was **interrupted at user direction**, retaining a full step150 checkpoint. There is no terminal V48 accuracy claim or queued follow-up pose evaluation. Active work is documented in `JEPA_RECOVERY_PRIORITY_20260926.md` and excludes pose/readout optimization.
