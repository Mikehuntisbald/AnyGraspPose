"""Explicit v1 -> single or trained single -> dual warm-start, never optimizer resume."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import migrate,save_init

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',default=os.environ.get('LIP_INIT_CKPT'))
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not a.checkpoint:p.error('Set LIP_INIT_CKPT or --checkpoint; no checkpoint is selected by timestamp')
    config=load_stream_config(a.config);model=make_model(config)
    report=migrate(model,a.checkpoint);save_init(a.out,model,config,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('parameters','source_skipped','parent')},indent=2))

if __name__=='__main__':main()
