# Shared recovery gradient audit V51

No optimizer updates or pose evaluation. V44 source,32 distinct training observations,64 paired hypotheses,8 independent batches.

|Gradient site|CE / geometry norm minimum|Maximum|Median|Depth–CE cosine minimum|Maximum|
|---|---:|---:|---:|---:|---:|
|patch|7.779|15.263|11.282|-0.083|0.231|
|dense|5.590|9.127|7.635|-0.036|0.046|
|dpt_shared|3.625|9.934|4.573|-0.186|0.249|
|atlas_query|322.868|1081.202|753.123|0.000|0.000|
|dpt_output|0.000|0.000|0.000|0.000|0.000|

Shared recovery gradients are imbalanced on these batches; CE and depth mostly orthogonal, not universally opposed. This is not yet a causal generalization result. Test only correspondence weight1.0 vs0.1 with matched initialization/data200 updates.

Canonical CAD normal loss remains disabled. Final depth output parameters have zero correspondence gradient because correspondence reads the shared DPT features before the output convolution; the shared patch/DPT features do receive both gradients. Sparse256-query coverage is recorded per region but is not established as a cause.
