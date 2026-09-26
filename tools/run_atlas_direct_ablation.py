"""Test learned-only retrieval for both completed V43 arms, GPUs serially."""
import json,subprocess,sys
from pathlib import Path


def main():
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/atlas_direct_v43')
    if not json.loads((root/'status.json').read_text())['completed']:raise RuntimeError('Training not complete')
    for arm in ('assisted','direct'):
        subprocess.run([sys.executable,'tools/run_atlas_ablation.py','--config',f'configs/jepa/atlas_direct_v43_{arm}.yaml','--checkpoint',str(root/arm/'runs/seed42/last.pt'),'--out',str(root/arm/'ablation'),'--arms','learned_only'],check=True)
    (root/'ablation_status.json').write_text(json.dumps(dict(completed=True,training=False,models=2,records_each=64),indent=2)+'\n')


if __name__=='__main__':main()
