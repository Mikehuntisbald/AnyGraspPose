# V66: geometry feedback has useful but limited causal effect

Frozen V60-extra, 64 physical-holdout frames inside the training partition, seed42, eight GPUs. No training, pose evaluation, official test or default promotion. The same samples and region masks are used for all interventions; all first-round predictions are invariant, and the normal forward matches the baseline bitwise.

At 10-degree initial error and requested heavy occlusion, real-hidden second-round flow EPE is 5.8033px normally, 6.6862px with geometry feedback disabled, 5.4324px with oracle canonical XYZ, and 5.2285px with oracle XYZ and validity. Geometry feedback therefore contributes to flow on these samples.

At 60-degree error and heavy occlusion, real-hidden EPE is 27.7115px normally versus 27.1641px with oracle XYZ. Even correct dense XYZ passed through the current feedback mechanism does not solve large-error hidden matching. At 10-degree heavy CAD-proxy only two eligible frames exist, and the normal feedback worsens EPE from the feedback-off 1.6940px to 2.6687px; do not omit this adverse small slice or generalize it broadly.

Oracle XYZ replaces only GT-render-valid pixels, retaining predicted background XYZ and the original 14px average pooling. Oracle validity uses foreground/background logits +12/-12; the current feedback trust remains capped at 0.5. These are diagnostic interventions, not deployable predictions, not a ceiling on a redesigned matcher, and not proof of improved learned reconstruction. Current pooling can mix foreground/background or distinct surfaces; whether this limits matching requires a separate controlled test.

Together with V61's weak flow-to-reconstruction effect, the evidence motivates redesigning spatial evidence transport and testing surface-preserving feedback rather than only scaling losses. Continue to preserve measured RGB-D and independent correspondence/geometry targets. Never treat estimated-render depth as recovered camera depth.

See REPORT.md for every condition, rank*/frames.jsonl for paired raw results, source_receipt.json for executed source hashes, and rank*/receipt.json for checkpoint identity and intervention scope. The immutable remote runtime is /tmp/dexycb_geometry_feedback_v66_r0. The remote repository directory is a runtime copy, not a Git checkout.
