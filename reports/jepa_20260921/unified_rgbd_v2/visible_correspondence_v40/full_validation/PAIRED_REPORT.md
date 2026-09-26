# V40 fixed40 outcome

|Checkpoint|Real sensor XYZ mm|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|---:|
|source|38.539|37.071|13.361|36.735|19.883|
|corrupted|38.939|37.461|13.725|37.709|20.838|
|clean|38.511|36.994|14.389|37.349|20.807|

Both terminal models completed100 updates. No geometry improvement on this full protocol; source remains the reference. These are reconstruction metrics under shared baseline-conditioned crops, not native pose scores. History is disabled.

The separate base-pose audit is descriptive: canonical real XYZ is7.94mm in the <=15deg bin and76.45mm above45deg. Populations differ. Geodesic angles are not symmetry reduced; this does not by itself prove unavailable CAD surface is the cause.
