"""Audit completed paired training receipts and bounded geometry CE."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    status=json.loads((root/'status.json').read_text())
    assert status['stage']=='complete' and status['completed']
    rows={};receipts={};values=[]
    for arm in ('control','surface'):
        folder=root/arm/'seed42'
        receipt=json.loads((folder/'last.receipt.json').read_text())
        assert receipt['completed'] and receipt['step']==200 and receipt['all_rank_rng']==8
        restore=json.loads((folder/'resume2.json').read_text())
        assert restore['complete_state_verified'] and restore['step']==2
        receipts[arm]=receipt
        for rank in range(8):
            records=[json.loads(line) for line in (folder/f'rank{rank}.jsonl').read_text().splitlines()]
            assert [r['step'] for r in records]==list(range(1,201))
            for r in records:
                assert math.isfinite(r['loss']) and math.isfinite(r['grad_norm'])
                ce=[v for k,v in r['metrics'].items() if 'recovery_ce' in k]
                assert not ce
            integrity=json.loads((folder/f'flow_integrity_rank{rank}.json').read_text())
            assert integrity['teacher_input'] is False
            assert integrity['native_crop_max_error_px']<.002
            for key in ('reconstruction_to_flow','flow_to_recovered_xyz'):
                assert math.isfinite(integrity[key]) and integrity[key]>0
            rows[arm,rank]=records
    for rank in range(8):
        for a,b in zip(rows['control',rank],rows['surface',rank]):
            assert a['windows']==b['windows'],(rank,a['step'])
            assert a['lr']==b['lr'],(rank,a['step'])
    result=dict(completed=True,updates_per_arm=200,ranks=8,
                matching_window_metadata_and_learning_rates=True,
                note='Window metadata equality does not independently hash every input pixel.',
                all_losses_gradients_finite=True,complete_state_resume_receipts_verified=True,
                historical_control_replay=json.loads((root/'control_replay_exact.json').read_text()),checkpoint_receipts=receipts)
    (root/'training_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
