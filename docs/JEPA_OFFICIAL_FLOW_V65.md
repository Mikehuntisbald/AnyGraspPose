# V65: official pretrained flow reference, not a deployed replacement

Use the [official GoTrack source](https://github.com/facebookresearch/gotrack/tree/68f76055755f2a4a8967e13ece834f975f008bdf)
(commit68f76055755f2a4a8967e13ece834f975f008bdf, CC BY-NC4.0), including its pinned
DINOv2 submodule12592a9bdfea0a6e4ee4767cd5a8d3195c0e3509. ModelScope indexed search
returned no matching model; this does not prove no mirror exists. The official
Git-LFS checkpoint is1608850339 bytes, SHA256
`f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086`.

The existing Python/PyTorch environment is retained. The adapter imports the
unmodified official DINO extractor,12-layer decoder and DPT flow/confidence head.
It disables both automatic backbone network downloads and loads all532 tensors
strictly from `model_state_dict`, after requiring the `models.1.` namespace.
No missing tensors or random backbone weights are accepted. The complete official
checkpoint remains unchanged. A direct execution of the official compute_flow
and extract_features methods matches adapter outputs bitwise on the checked pair.

Collect96 physical-holdout training-partition frames,32 each at0/10/60-degree
controlled references, requested natural/light/heavy conditions. Cache raw
corrupted student RGB, estimated-pose CAD RGB and template mask separately from
labels. The official worker reads only student inputs and emits dense flow and
visibility. It never invokes PnP or uses GT overlays/labels. Compare with the V60
local-flow model on the same source points and original unfiltered region masks.

RGB is resized224->280 with align_corners=False. Predicted displacements are
sampled using matching pixel-center coordinates and divided by1.25; interpolation
is normalized by valid template-mask mass. Zero-coverage points remain in the
metric denominator. Unit controls verify identity, translation and mask mass.
Shared-RGB and official0.5-gray-background variants are retained; outputs were
identical on all96 frames. This uses our crop/render protocol, not GoTrack's full
native preprocessing or published benchmark; training budgets and modalities
(RGB versus RGB-D) also differ.

Additional checks include eight fixed CAD-only identity and7,-5px translation
pairs, the official0.3 visibility threshold as a conditional diagnostic, native
RGB/depth/pose-cache equality for a fixed moving-object case, and pose-cache frame
indices across all6400 training streams. No confidence-based target filtering,
label shifting, default promotion, training or official-test access occurs.

A fitted fractional temporal offset on one blurred RGB case is a diagnostic
hypothesis, not a calibrated delay or permission to alter ground truth. The
[DexYCB paper](https://arxiv.org/abs/2104.04631) describes synchronized multi-view
capture; the present evidence cannot establish a dataset-wide synchronization
fault. Motion blur, annotation/calibration accuracy and learned matching errors
remain possible contributors.
