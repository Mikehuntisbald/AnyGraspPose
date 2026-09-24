# LR-only paired replay

Preserved main checkpoint step26500 SHA256 `27437eb68f6b31c145378514b36d1e48a04661f0493d5f21a5cfebad4f0b2757`. Old controller interrupted; at most49 unsaved updates may need replay, complete checkpoint progress retained.

Replay step25400 to26400 with scale0.3 of the same global200-step rewarm /10000-step cosine. Reuse completed scale1.0 evaluation at26400. Same initial weights, Adam moments/steps, EMA, all-rank RNG, sampled episodes, loss and8GPU batch32. Only LR values change; schedule clock remains25400..35400.

Select0.3 if mean ratio of eight light/heavy real/proxy XYZ/depth errors <=0.99, all ratios <=1.03, and all four retrieval differences >=-0.005. Otherwise retain1.0; this bounded search does not establish an optimal LR or prove that LR caused all prior regression. Validation is explicitly used for selection, not independent confirmation. Training loss and cosine alone do not select the winner.

Resume selected LR from preserved26500, not the replay checkpoint. Final target35400 unchanged. Evaluate30400/35400. No architecture/loss/encoder reset, no history/pose training, no official test. Probe adds1000 diagnostic updates to compute budget, not final-model step count.
