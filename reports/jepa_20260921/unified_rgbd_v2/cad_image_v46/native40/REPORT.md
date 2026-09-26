# Fixed40 conditional native-frame correspondence evaluation

Previous-frame sealed LIP estimates define the common crop/base.119 frames and238 natural/heavy records per model. No GT student input; no closed-loop claim. Raw rotation is not symmetry reduced.

|Model/input|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|
|---|---:|---:|---:|---:|
|source/natural|—|—|40.429|15.840|
|source/heavy|34.717|11.616|40.376|17.064|
|parent/natural|—|—|39.888|18.739|
|parent/heavy|37.313|13.794|40.817|19.711|
|trained/natural|—|—|40.373|18.336|
|trained/heavy|37.182|14.196|41.622|19.310|

|Model/input/readout|ADD-S mm|ADD-S<0.05d %|Accepted %|
|---|---:|---:|---:|
|source/natural/base|11.734|67.083|—|
|source/natural/frozen_head|11.440|69.583|—|
|source/natural/image_to_cad_rgbd_mixed|11.993|66.250|95.8|
|source/heavy/base|11.734|67.083|—|
|source/heavy/frozen_head|11.690|68.750|—|
|source/heavy/image_to_cad_rgbd_mixed|12.158|63.333|97.5|
|parent/natural/base|11.734|67.083|—|
|parent/natural/frozen_head|11.420|69.167|—|
|parent/natural/cad_to_image|74.309|49.583|23.3|
|parent/natural/cad_to_image_visible|171.409|30.833|52.5|
|parent/natural/cad_to_image_rgbd_mixed|12.195|61.667|30.0|
|parent/natural/image_to_cad_rgbd_mixed|12.248|61.667|85.0|
|parent/heavy/base|11.734|67.083|—|
|parent/heavy/frozen_head|11.826|70.000|—|
|parent/heavy/cad_to_image|89.597|52.917|21.7|
|parent/heavy/cad_to_image_visible|176.137|25.833|54.2|
|parent/heavy/cad_to_image_rgbd_mixed|12.015|59.167|30.0|
|parent/heavy/image_to_cad_rgbd_mixed|12.501|57.083|75.0|
|trained/natural/base|11.734|67.083|—|
|trained/natural/frozen_head|11.824|67.083|—|
|trained/natural/cad_to_image|53.214|34.167|47.9|
|trained/natural/cad_to_image_visible|106.375|14.167|77.5|
|trained/natural/cad_to_image_rgbd_mixed|11.727|66.250|34.2|
|trained/natural/image_to_cad_rgbd_mixed|11.987|60.417|90.8|
|trained/heavy/base|11.734|67.083|—|
|trained/heavy/frozen_head|11.987|68.750|—|
|trained/heavy/cad_to_image|49.815|34.167|50.4|
|trained/heavy/cad_to_image_visible|56.339|31.250|47.5|
|trained/heavy/cad_to_image_rgbd_mixed|11.850|64.167|42.5|
|trained/heavy/image_to_cad_rgbd_mixed|12.795|52.500|90.0|
