"""Compose unmodified official GoTrack modules for flow-only evaluation.

Vendor source: facebookresearch/gotrack68f76055755f2a4a8967e13ece834f975f008bdf,
CC BY-NC4.0. No pose solver, detector, extra environment or network model lookup.
Run in a separate process from the LIP model to isolate generic vendor imports.
"""
import sys,hashlib,json
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
ROOT=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/official_flow_v65')
SOURCE=ROOT/'source'
WEIGHT_SHA='f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'
sys.path.insert(0,str(SOURCE/'external/dinov2'))
sys.path.insert(0,str(SOURCE))


class OfficialFlow(nn.Module):
    def __init__(self):
        super().__init__()
        import dinov2.hub.backbones as backbones
        from utils.dinov2_util import DinoFeatureExtractor
        from model.blocks.decoder import Decoder
        from model.blocks.config import DecoderOpts
        from model.heads.dpt.model import DPTHead
        from model.heads.dpt.config import DPTHeadOpts
        factory=backbones.dinov2_vits14_reg
        def make(*args,**kwargs):return factory(pretrained=False)
        # The official extractor redundantly constructs its backbone twice.
        # Disable both network lookups: all tensors must come from the checked LFS weight.
        with patch.object(backbones,'dinov2_vits14_reg',make),patch.object(torch.hub,'load',make):
            self.backbone=DinoFeatureExtractor('dinov2_vits14-reg')
        self.decoder=Decoder(DecoderOpts())
        self.pose_head=DPTHead(DPTHeadOpts())

    def forward(self,query,template,mask):
        assert query.shape==template.shape and query.shape[1:]==(3,280,280)
        assert mask.shape==(len(query),280,280)
        features=self.backbone(torch.cat((query,template),0))['feature_maps']
        q,t=features.chunk(2,0)
        _,reference=self.decoder(features_query=q,features_reference=t,crop_size=(280,280))
        flow,confidence=self.pose_head(reference,(280,280))
        return flow*mask[:,None],confidence*mask


def load(device='cuda'):
    weight=ROOT/'weights/gotrack.pt'
    assert hashlib.file_digest(weight.open('rb'),'sha256').hexdigest()==WEIGHT_SHA
    data=torch.load(weight,map_location='cpu',weights_only=True)
    state=data['model_state_dict']
    assert all(k.startswith('models.1.') for k in state)
    state={k.removeprefix('models.1.'):v for k,v in state.items()}
    model=OfficialFlow();model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False)


if __name__=='__main__':
    torch.set_num_threads(2)
    model=load()
    with torch.no_grad():
        image=torch.full((1,3,280,280),.5,device='cuda')
        flow,confidence=model(image,image,torch.ones(1,280,280,device='cuda'))
    assert torch.isfinite(flow).all() and torch.isfinite(confidence).all()
    receipt=dict(strict_state_load=True,sha256=WEIGHT_SHA,flow_shape=list(flow.shape),confidence_shape=list(confidence.shape),
        feature_layer=model.backbone.layer,backbone=model.backbone.model_base_name,modules_imported_from=str(SOURCE),
        flow_only=True,pose_solver_called=False,network_backbone_lookup=False)
    (ROOT/'adapter_smoke.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
