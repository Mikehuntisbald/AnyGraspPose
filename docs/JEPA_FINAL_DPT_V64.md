# V63/V64: make completed JEPA latents feed every DPT scale

V63 audits64 frozen V60-extra frames from physical training holdout. Shifting
one DPT input horizontally by one patch changes output geometry, including the
last input. It is therefore false to say the last level is unused. But its mean
input-norm-scaled gradient share is only2.43% for depth and5.71% for the atlas XYZ
surrogate; the first two levels account for88.45% and81.43%. These are local
sensitivity measurements, not percentages of total information or causal credit.

V64 changes the DPT input contract: every invocation reassembles all four scales
from that stage's latest JEPA latent. Earlier states remain in JEPA's computation
but no longer directly feed the geometry decoder. This applies to intermediate
recovery and final recovery, keeping the shared decoder's contract consistent.
No external encoder/pose bypass, new trainable parameters, or teacher input is
introduced. All old tensors load exactly; the optional flag defaults to false.

Candidate starts from the same V60-extra export as V62, with100x flow write,
identical optimizer, seeds, paired/zero/empirical initialization curriculum and
200-step budget. The matched historical control is V62 strong200. Before reuse,
this runtime replays its first2 updates on all8 ranks and requires exact loss,
gradient norm, learning rates, metrics and sampled-window records. It does not
claim that all200 updates were rerun. Candidate separately stops after2 and
strictly resumes to200 with full optimizer/scheduler/RNG state.

Fixed paired probes:32 correct-reference,64 ten-degree,32 sixty-degree observations,
seed62050000, same original real/CAD-proxy target masks as V62. No native validation,
official test, extra seed or automatic default promotion. A short negative trial
would not prove that final-only decoding can never work, but cannot justify a
geometry-accuracy claim or blind budget extension.
