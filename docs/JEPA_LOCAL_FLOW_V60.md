# V60: local flow input/optimization control

V59 found that 10-degree heavy real-hidden flow improves from7.896px to5.766px with reconstruction feedback, but coarse5.778px barely changes under the local head. Only2.3% lie outside its per-axis14px correction range. At60degrees,71.2% lie outside that range. The small-error readout and large-error coarse matcher need separate remedies.

This stage preserves the coarse matcher, write module, encoder, JEPA, DPT and full CAD atlas at frozen V56 source044a8f...0fa5f. It tests whether explicit coordinate/matching evidence helps the local endpoint head. It is not a claim of final geometry accuracy or a dense GoTrack reproduction.

Both arms have the same210->128->2 network. The first133 inputs exactly match the original endpoint head. Both start with all old endpoint weights copied and new input weights zero. Output remains a14px tanh correction; original support/visibility heads remain unchanged. The control zeros the extra77 inputs. The candidate receives normalized coarse UV, coarse-source displacement,3x3 relative correlation scores, and sampled real observation descriptor. No GT enters the student.

Eight GPUs cache both original frozen rounds on2048 training frames and256 physically disjoint heldout frames in the training partition. Both heads receive identical1024-point minibatches for1000 updates, AdamW1e-3, clip1, pixel SmoothL1 beta1. Sampling balances each eligible frame/region; CAD-proxy weight0.5. Holdout is not used for checkpoint selection. A2-step stop/resume verifies weights, optimizer and CPU/CUDA RNG exactly.

The cached evaluation does not include feedback distribution shifts. Therefore final exported models run both rounds from raw observations on64 new10-degree and32 new60-degree cases, against the same frozen baseline. Report flow and unfiltered source-owned canonicalXYZ/depth together. A flow improvement alone cannot satisfy the geometry objective. Head-stage checkpoints are fully resumable; full-model exports are inference-only and must not be passed directly to the geometry trainer.

No pose training, history, official test, default-model promotion or automatic extra budget. Full8192 CAD assets and real-depth target ownership remain unchanged.
