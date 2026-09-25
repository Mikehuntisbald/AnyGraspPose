import os,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from unified_val_guard import UnifiedNativeValGuard


def test_only_static_metadata_is_allowed(tmp_path):
    raw=tmp_path/'raw';index=tmp_path/'index';index.mkdir();(index/'streams.jsonl').write_text('')
    metadata=raw/'bop/models/models_info.json'
    guard=UnifiedNativeValGuard(raw,index,tmp_path/'fp',[],cad_metadata=metadata)
    guard.check(metadata)
    assert guard.counts['static_cad_metadata_reads']==1
    for p in (raw/'bop/train/000001/scene_gt.json',raw/'bop/val/000001/scene_gt.json',raw/'bop/models/unpinned.json'):
        with pytest.raises(PermissionError):guard.check(p)
    with pytest.raises(PermissionError):guard.check(metadata,os.O_WRONLY)
    with pytest.raises(PermissionError):guard.check(metadata,native=True)
