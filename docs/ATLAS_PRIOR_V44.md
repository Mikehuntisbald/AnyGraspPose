# V44 final coordinate-prior supervision

V43 learned-only correspondence improved when CE lost access to the coordinate prior, but the final DPT XYZ prior worsened and its direct output gradient was exactly zero. Coarse DPT supervision does not directly supervise the refined final-pass XYZ, despite shared head weights.

Two100-update arms start from V43 direct100 (SHA2567bd2ae55c8172467ae4c4990928b604ec0d4536e7179812ca34b9c17659b8246), with identical fresh optimizer/RNG and train sampler45000000. Both keep prior-free correspondence CE, the same inference ranking, audited real-depth masks and temporary normal exclusion. Control adds no loss; supervised adds final DPT XYZ SmoothL1 with the same real/proxy/visible canonical targets, weights1/0.5/0.5, and scalar weight1. There is no duplicate depth/validity term and no GT in the forward path.

A regression test proves the previous final XYZ gradient is zero, the added term supplies correctly signed gradient only on eligible pixels, and depth/validity gradients are unchanged. The first full-model batch records this gradient on every rank and refuses to proceed if supervised XYZ gets none.26 targeted tests passed in the launcher. Each arm uses8 H20s and32 distinct observations/64 pose hypotheses per update, strict2→100 resume and unchanged64-record corrupted/clean probes. No history, pose loss, multi-seed or official test.

Runtime `/tmp/dexycb_atlas_prior_v44`; artifact root `/mnt/why/dexycb_lip/unified_jepa_20260921/atlas_prior_v44`. No result or promotion is assumed before paired evaluation completes.
