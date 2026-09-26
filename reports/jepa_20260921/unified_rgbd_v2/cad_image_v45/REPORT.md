# V45 explicit CAD-to-image outcome

All numbers below use the heavy subset and equal physical-sequence means. Controlled64 probes, not native tracking. Original target masks are unchanged.

|Checkpoint|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|0|14.422|12.505|18.336|13.360|
|100|13.376|12.422|15.561|12.826|
|source|9.482|9.871|14.996|13.473|
|initial|14.422|12.505|18.336|13.360|
|trained|13.376|12.422|15.561|12.826|

|Checkpoint/readout|Rotation deg|Translation mm|ADD-S mm|ADD-S<0.05d %|Accepted %|
|---|---:|---:|---:|---:|---:|
|0/base|10.000|0.000|5.595|100.000|—|
|0/cad_to_image|13.624|61.781|54.844|77.586|22.4|
|0/frozen_head|10.129|3.019|5.874|100.000|—|
|0/image_to_cad|13.680|13.644|11.498|68.966|31.0|
|100/base|10.000|0.000|5.595|100.000|—|
|100/cad_to_image|14.593|72.576|61.462|68.966|31.0|
|100/frozen_head|9.830|2.935|5.704|100.000|—|
|100/image_to_cad|14.516|17.797|13.177|62.069|48.3|
|source/base|10.000|0.000|5.595|100.000|—|
|source/frozen_head|10.286|2.867|5.968|93.103|—|
|source/image_to_cad|11.266|14.895|9.514|51.724|93.1|
|source/image_to_cad_visible|10.009|3.491|6.703|93.103|17.2|
|initial/base|10.000|0.000|5.595|100.000|—|
|initial/cad_to_image|13.624|61.781|54.844|77.586|22.4|
|initial/cad_to_image_prior16|14.044|135.845|95.591|10.345|96.6|
|initial/cad_to_image_prior32|27.415|302.119|255.068|12.069|87.9|
|initial/cad_to_image_visible|39.257|409.897|365.178|31.034|69.0|
|initial/frozen_head|10.129|3.019|5.874|100.000|—|
|initial/image_to_cad|13.680|13.644|11.498|68.966|31.0|
|initial/image_to_cad_visible|9.901|5.054|8.138|94.828|5.2|
|trained/base|10.000|0.000|5.595|100.000|—|
|trained/cad_to_image|14.593|72.576|61.462|68.966|31.0|
|trained/cad_to_image_prior16|10.000|0.000|5.595|100.000|0.0|
|trained/cad_to_image_prior32|10.000|0.000|5.595|100.000|0.0|
|trained/cad_to_image_visible|10.000|0.000|5.595|100.000|0.0|
|trained/frozen_head|9.830|2.935|5.704|100.000|—|
|trained/image_to_cad|14.516|17.797|13.177|62.069|48.3|
|trained/image_to_cad_visible|10.957|6.280|8.429|94.828|10.3|

Poses rejected by prediction-only checks retain the base pose and remain in metrics. Base translation is exactly correct in this protocol; ADD-S success is already saturated. Endpoint, continuous pose error, acceptance, and wider-initialization/native tests are needed.
