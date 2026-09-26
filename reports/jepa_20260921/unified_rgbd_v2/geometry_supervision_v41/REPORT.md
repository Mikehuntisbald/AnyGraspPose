# V41 audited supervision — matched100 updates

|Model|Input|Real CAD XYZ mm|Real sensor XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---|---:|---:|---:|---:|---:|
|v41_0|corrupted|9.482|14.762|9.871|14.996|13.473|
|v41_0|clean|9.537|14.682|8.266|15.228|13.056|
|v40_100|corrupted|9.715|14.075|9.891|11.448|10.192|
|v40_100|clean|9.621|13.941|8.046|10.849|7.863|
|v41_100|corrupted|9.477|14.306|11.561|12.783|13.324|
|v41_100|clean|9.839|14.661|9.414|12.096|11.021|

Original evaluation masks retained; no evaluation filtering. Same initialization, optimizer, scheduler, eight rank RNG states, sampler, observations and100-update budget. Changed training supervision quality and FP32 geometry operations. Clean input is a privileged availability diagnostic, not heavy-occlusion deployment.
