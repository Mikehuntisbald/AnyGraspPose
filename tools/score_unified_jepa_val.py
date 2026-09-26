"""Preserve native LIP scoring, aggregate unified backbone shard counters."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.object_jepa_checkpoint import atomic_json
from lip.engine.jepa_checkpoint import sha

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--world',type=int,required=True)
    a,_=p.parse_known_args();shards=[json.loads((a.run/f'rank{i}/manifest.json').read_text()) for i in range(a.world)]
    for m in shards:
        if m['architecture_id'] not in ('stream_dino_utonia_jepa_rgbd_v2','stream_dino_fp_staticutonia_jepa_rgbd_v3','stream_two_input_jepa_v9','stream_conv_cross_jepa_v10','stream_conv_cross_geohistory_jepa_v10','stream_conv_cross_supported_history_jepa_v10','stream_conv_cross_dense_history_jepa_v10','stream_recovered_relation_jepa_v11','stream_cad_surface_jepa_v12','stream_serial_completion_jepa_v21'):raise ValueError('Wrong architecture')
        for key in ['architecture_id','jepa_enabled','history_enabled','checkpoint_sha256','stage_optimizer_steps','config_sha256']:
            if m[key]!=shards[0][key]:raise ValueError('Inconsistent shard identity: '+key)
        for key in ('disable_shared_patch','shared_patch_enabled'):
            if m.get(key)!=shards[0].get(key):raise ValueError('Inconsistent readout intervention: '+key)
    import score_val_non_gt
    score_val_non_gt.main()
    result=json.loads((a.out/'manifest.json').read_text())
    for key in ['current_dino_images','rendered_dino_images','rejected_updates']:result[key]=sum(m[key] for m in shards)
    for key in ['max_memory_tokens','max_history_frames']:result[key]=max(m[key] for m in shards)
    result['seconds']=max(m['seconds'] for m in shards)
    result['timing_scope']='Maximum shard loop; excludes setup/scoring'
    result['scoring_entrypoint_sha256']=sha(__file__);result['unified_jepa_counters_aggregated']=True
    atomic_json(a.out/'manifest.json',result)

if __name__=='__main__':main()
