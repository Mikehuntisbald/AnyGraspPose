# Explicit CAD-to-image outcome: cad_image_v47

All numbers below use the heavy subset and equal physical-sequence means. Controlled64 probes, not native tracking. Original target masks are unchanged.

|Checkpoint|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|0|14.318|13.337|16.084|13.978|
|500|14.318|13.337|16.084|13.978|

|Checkpoint/readout|Rotation deg|Translation mm|ADD-S mm|ADD-S<0.05d %|Accepted %|
|---|---:|---:|---:|---:|---:|
|0/anchored_flow|10.000|0.000|5.595|100.000|94.8|
|0/anchored_flow_rgbd_measured|11.895|6.076|7.206|72.414|37.9|
|0/anchored_flow_rgbd_mixed|13.420|9.961|8.630|53.448|82.8|
|0/anchored_flow_rgbd_recovered|14.398|10.575|8.941|41.379|82.8|
|0/base|10.000|0.000|5.595|100.000|—|
|0/cad_to_image|14.520|56.035|39.190|58.621|46.6|
|0/cad_to_image_rgbd_measured|13.950|6.402|8.163|82.759|25.9|
|0/cad_to_image_rgbd_mixed|12.009|4.318|6.921|75.862|48.3|
|0/cad_to_image_rgbd_recovered|12.394|5.420|7.340|60.345|55.2|
|0/cad_to_image_visible|15.577|76.951|57.627|39.655|62.1|
|0/frozen_head|9.645|4.257|6.006|96.552|—|
|0/image_to_cad|10.000|0.000|5.595|100.000|0.0|
|0/image_to_cad_rgbd_measured|10.320|2.503|6.390|89.655|15.5|
|0/image_to_cad_rgbd_mixed|13.559|7.818|7.735|67.241|79.3|
|0/image_to_cad_rgbd_recovered|13.424|8.813|8.195|58.621|86.2|
|0/image_to_cad_visible|11.073|28.294|27.933|87.931|12.1|
|500/anchored_flow|11.937|41.656|22.668|13.793|87.9|
|500/anchored_flow_rgbd_measured|10.780|5.801|7.546|72.414|34.5|
|500/anchored_flow_rgbd_mixed|14.633|10.266|9.159|53.448|86.2|
|500/anchored_flow_rgbd_recovered|14.827|11.202|9.198|41.379|89.7|
|500/base|10.000|0.000|5.595|100.000|—|
|500/cad_to_image|14.520|56.035|39.190|58.621|46.6|
|500/cad_to_image_rgbd_measured|13.950|6.402|8.163|82.759|25.9|
|500/cad_to_image_rgbd_mixed|12.009|4.318|6.921|75.862|48.3|
|500/cad_to_image_rgbd_recovered|12.394|5.420|7.340|60.345|55.2|
|500/cad_to_image_visible|15.577|76.951|57.627|39.655|62.1|
|500/frozen_head|9.645|4.257|6.006|96.552|—|
|500/image_to_cad|10.000|0.000|5.595|100.000|0.0|
|500/image_to_cad_rgbd_measured|10.320|2.503|6.390|89.655|15.5|
|500/image_to_cad_rgbd_mixed|13.559|7.818|7.735|67.241|79.3|
|500/image_to_cad_rgbd_recovered|13.424|8.813|8.195|58.621|86.2|
|500/image_to_cad_visible|11.073|28.294|27.933|87.931|12.1|

Poses rejected by prediction-only checks retain the base pose and remain in metrics. Base translation is exactly correct in this protocol; ADD-S success is already saturated. Endpoint, continuous pose error, acceptance, and wider-initialization/native tests are needed.
