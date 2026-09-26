# V43 matched direct correspondence supervision

|Arm/step|Input|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---|---:|---:|---:|---:|
|assisted_0|corrupted|13.313|11.143|16.292|12.889|
|assisted_0|clean|13.438|9.794|17.738|13.734|
|assisted_100|corrupted|13.932|11.948|16.775|12.933|
|assisted_100|clean|13.739|9.797|18.152|14.169|
|direct_100|corrupted|15.278|12.592|18.757|12.970|
|direct_100|clean|14.700|10.855|20.047|13.932|

Same inference and original labels; only coordinate-prior term in correspondence CE differs. No pose/native/history claim. Frozen learned-only intervention remains necessary to assess the intended matching mechanism.
