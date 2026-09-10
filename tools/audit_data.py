import argparse,json,os,pathlib
from lip.data.index import SUBJECTS,CAMERAS
from lip.engine.config import environment
p=argparse.ArgumentParser();p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'));p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
r=dict(environment=environment(),data_root=a.data_root,configured=bool(a.data_root),units_status='unverified until check_geometry',raw_read_only=True)
if a.data_root:
 root=pathlib.Path(a.data_root)
 r['subjects']={s:len(list((root/s).glob('*/meta.yml'))) for s in SUBJECTS}
 r['calibration_present']=(root/'calibration'/'intrinsics').is_dir();r['models_present']=(root/'models').is_dir()
 r['complete_inventory']=all(n==100 for n in r['subjects'].values()) and r['calibration_present'] and r['models_present']
(out/'audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if not r.get('complete_inventory'):raise SystemExit(2)
