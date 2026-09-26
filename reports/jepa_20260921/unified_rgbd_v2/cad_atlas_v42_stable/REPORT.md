# V42 complete CAD hard-point decoder

|Model|Input|Real CAD XYZ mm|Real sensor XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---|---:|---:|---:|---:|---:|
|source_0|corrupted|9.482|14.762|9.871|14.996|13.473|
|source_0|clean|9.537|14.682|8.266|15.228|13.056|
|v41_100|corrupted|9.477|14.306|11.561|12.783|13.324|
|v41_100|clean|9.839|14.661|9.414|12.096|11.021|
|v42_0|corrupted|17.312|19.005|12.213|21.888|16.814|
|v42_0|clean|17.738|18.245|11.644|21.863|18.740|
|v42_100|corrupted|13.313|17.185|11.143|16.292|12.889|
|v42_100|clean|13.438|16.537|9.794|17.738|13.734|

All64 records and original evaluation masks match. Same parent tensors; V42 adds a random full-CAD query/key head. Interrupted normal-loss attempt preserved separately; stable restart has identical initial model/optimizer/RNG and temporarily disables finite-difference normals. Empty-CAD fallback is DPT; no FP/pose shortcut. Hard membership in a CAD point cloud is not proof of correct correspondence.
