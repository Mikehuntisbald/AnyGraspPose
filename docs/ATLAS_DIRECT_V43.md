# V43 matched direct correspondence supervision

V42's frozen ablation found no aggregate benefit from learned similarity: heavy real/proxy CAD XYZ13.313/16.292mm with full retrieval,13.284/16.085mm using only the DPT coordinate prior, and37.741/53.746mm with learned similarity alone. All64 inputs, target masks and predicted depths match across the interventions. The initial ablation attempt failed from a probe-local function-name collision, before completing a result; its directory remains. The corrected ablation uses an explicitly captured score function in a new pinned runtime.

This motivates a specific paired training test, not an unqualified longer run. Two arms start from the exact V42 stable100 tensors (SHA256c1483df2bc2ae78b2ac4c929444d2b6bf9e6b991980c1b1f0f9015309368473f), identical fresh AdamW/RNG, seed42, sampler start43000000 and100 updates each. Both retain the same inference ranking (learned score plus DPT prior), geometry/depth/validity objectives, V41 teacher quality masks and temporary normal-loss exclusion.

Only the correspondence CE differs: assisted keeps the DPT coordinate-prior term in the logits; direct removes it so query/key matching must predict the same GT CAD point neighborhood without help from that term. It does not remove estimated-camera geometry from keys or positional information from JEPA; this is not a test of appearance alone. A unit test changes the coordinate prior arbitrarily and verifies direct CE is exactly unaffected.25 tests pass before startup. Source snapshots are immutable during both arms.

Each arm performs2 updates, corrupted/clean probe0, full2→100 resume, then identical probe100. Original64-record evaluation masks remain unfiltered. Frozen learned-only ablation of both terminal checkpoints will be needed to determine whether direct CE improves the intended matching ability, independently of continued training.

Runtime `/tmp/dexycb_atlas_direct_v43`; artifacts `/mnt/why/dexycb_lip/unified_jepa_20260921/atlas_direct_v43`. No multi-seed, official test or automatic promotion. Full accurate geometry is still unproven.
