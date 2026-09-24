# V18 continuation: 25400 to 35400

Authorized: 10000 additional JEPA-only updates, seed42, history off, pose frozen.
Source step25400 SHA256: `b202bc812230c6ecb7b9fd61f1a4e1d522dfa4fa848953880342842154a36879`.
Model, Adam moments/steps, EMA and sampler/RNG inherited. Architecture and losses unchanged.

Learning rates: inherited terminal encoder1e-7 / predictor1e-6 / new1e-5; 200-step linear rewarm to encoder1e-6 / predictor1e-5 / new1e-4; cosine decay over remaining9800 steps to0.1 peak. Peak is10x source terminal, equal to prior stage peak.

Checkpoint every50. Fixed40 recovery evaluations at26400/30400/35400. Terminal same-checkpoint RoPE on/off. Evaluation serially occupies8GPUs; no official test or default-model promotion. Compare with V18 step25400; no pose efficacy claims from recovery metrics.

Live state: `LIVE_STATUS.md`; terminal verification requires startup/terminal audits and collection receipt.

H20 preflight passed: full40-frame backward finite, exact Adam inheritance, EMA counter15650, pose/history gradients absent, discarded preflight update only. Controller launched for8 GPUs; startup resume audit passed at25403 on all8 ranks; inherited model and Adam state exact, frozen pose unchanged. Controller advanced to train_to_26400. Read-only local collection monitor active. Source/config commit90b2df5 pushed to origin/main. Remote isolated runtime copied from pinned V18 source archive, with extension config/controller; remote primary Git HEAD not asserted.
