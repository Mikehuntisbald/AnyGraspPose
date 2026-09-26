"""Verify the frozen-backbone experiment against actual terminal tensors."""
import argparse,json
from pathlib import Path
import torch
from lip.engine.jepa_checkpoint import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    run=a.root/'runs/seed42'
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert final['step']==500
    prefix='cad_atlas_decoder.image_readout.anchor_flow.'
    frozen=[n for n in initial['model'] if not n.startswith(prefix)]
    changed=[n for n in initial['model'] if not torch.equal(initial['model'][n],final['model'][n])]
    assert changed and all(n.startswith(prefix) for n in changed)
    assert all(torch.equal(initial['model'][n],final['model'][n]) for n in frozen)
    names=[n for group in final['optimizer']['param_groups'] for n in group['names']]
    assert names and all(n.startswith(prefix) for n in names)
    assert json.loads((run/'resume2.json').read_text())['complete_state_verified']
    receipt=dict(verified=True,step=500,backbone_frozen=True,frozen_parameters_and_buffers=len(frozen),changed_tensor_names=changed,
        trained_parameters=sum(initial['model'][n].numel() for n in names),optimizer_only_new_head=True,strict_resume=True,
        checkpoint_sha256=sha(run/'last.pt'),source_checkpoint_sha256=final['config']['geometry_transport_training']['source_sha256'],
        status_note='Original launcher backbone_frozen flag checks only decoder_only; this actual-tensor receipt supersedes that legacy field. Original runtime/status preserved.')
    (a.root/'frozen_readout_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
