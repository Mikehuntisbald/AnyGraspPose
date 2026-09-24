"""CPU-only derivation of local CAD tokens; no encoder or training updates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.cad_surface_cache import surface_tokens


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();c=yaml.safe_load(a.config.read_text());torch.set_num_threads(2)
    rows=[];begun=time.monotonic()
    for receipt in sorted(Path(c['paths']['cad_cache']).glob('*.json')):
        meta=json.loads(receipt.read_text())
        if meta.get('contract',{}).get('version')!='full-texture-utonia-v1':continue
        source=receipt.with_suffix('.pt')
        with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
        cad=torch.load(source,map_location='cpu',weights_only=True);cad.update(key=meta['key'],receipt=meta)
        assert cad['features'].shape==(8192,1224)
        before=hashlib.sha256(cad['features'].numpy().tobytes()).hexdigest()
        bank=surface_tokens(cad,c['cad_surface']['cache'],c['cad_surface']['tokens'])
        distance=torch.cdist(cad['coord'],bank['coord']).amin(-1)
        assert hashlib.sha256(cad['features'].numpy().tobytes()).hexdigest()==before
        rows.append(dict(parent_key=cad['key'],derived_key=cad['_surface_key'],points=len(bank['coord']),
                         feature_dim=bank['features'].shape[-1],coverage_mean_d=float(distance.mean()),coverage_max_d=float(distance.max())))
    assert rows
    result=dict(completed=True,device='cpu',optimizer_updates=0,original_caches_unchanged=True,seconds=time.monotonic()-begun,objects=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
