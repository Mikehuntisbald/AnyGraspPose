# Explicit CAD-to-image outcome: cad_image_v46

All numbers below use the heavy subset and equal physical-sequence means. Controlled64 probes, not native tracking. Original target masks are unchanged.

|Checkpoint|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|0|13.376|12.422|15.561|12.826|
|500|14.318|13.337|16.084|13.978|

|Checkpoint/readout|Rotation deg|Translation mm|ADD-S mm|ADD-S<0.05d %|Accepted %|
|---|---:|---:|---:|---:|---:|
|0/base|10.000|0.000|5.595|100.000|—|
|0/cad_to_image|14.593|72.576|61.462|68.966|31.0|
|0/cad_to_image_visible|10.000|0.000|5.595|100.000|0.0|
|0/frozen_head|9.830|2.935|5.704|100.000|—|
|0/image_to_cad|14.516|17.797|13.177|62.069|48.3|
|0/image_to_cad_visible|10.957|6.280|8.429|94.828|10.3|
|500/base|10.000|0.000|5.595|100.000|—|
|500/cad_to_image|14.520|56.035|39.190|58.621|46.6|
|500/cad_to_image_visible|15.577|76.951|57.627|39.655|62.1|
|500/frozen_head|9.645|4.257|6.006|96.552|—|
|500/image_to_cad|10.000|0.000|5.595|100.000|0.0|
|500/image_to_cad_visible|11.073|28.294|27.933|87.931|12.1|

Poses rejected by prediction-only checks retain the base pose and remain in metrics. Base translation is exactly correct in this protocol; ADD-S success is already saturated. Endpoint, continuous pose error, acceptance, and wider-initialization/native tests are needed.
