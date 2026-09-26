"""Teacher-only quarantine of RGB/depth/CAD disagreement, never relabel as CAD.

The threshold is an explicit research policy, not a calibration certificate.
Keep raw measurements and the historical scoring protocol intact.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F


@torch.no_grad()
def quarantine_real_geometry(target, scenes, visible_pixels, max_depth_gap_m):
    if not 0 < max_depth_gap_m <= .05:
        raise ValueError('Explicit positive depth disagreement threshold <= 50 mm required')
    if target.cad_geometry_depth_m is None or target.cad_geometry_valid is None:
        raise ValueError('Separate GT CAD metadata required for teacher-only audit')
    depth=torch.cat([s.depth for s in scenes])
    measured=torch.isfinite(depth)&(depth>0)
    cad=target.cad_geometry_valid
    agree=(depth-target.cad_geometry_depth_m).abs()<=max_depth_gap_m
    erode=lambda x:-F.max_pool2d(-x.float(),5,1,2)>.999
    eligible=erode(measured&cad&agree)
    real=target.geometry_real_weight&eligible
    # Disagreement is unknown. It is neither a negative surface label nor a CAD
    # replacement for an originally visible real-depth target.
    unknown=visible_pixels&measured&(~cad|~agree)
    label=target.geometry_valid_label.masked_fill(unknown,float('nan'))
    return replace(target,geometry_real_weight=real,geometry_visible_weight=real,
                   geometry_weight=real|target.geometry_proxy_weight,
                   geometry_valid_label=label,real_geometry_eligible=eligible)
