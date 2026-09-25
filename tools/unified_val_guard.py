"""Permit only pinned static CAD symmetry metadata beyond native-val image access."""
import os
from pathlib import Path
from val_non_gt_common import NativeValGuard

class UnifiedNativeValGuard(NativeValGuard):
    def __init__(self,*args,cad_metadata=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.cad_metadata=Path(cad_metadata).resolve() if cad_metadata else None
        if self.cad_metadata is not None and self.cad_metadata!=self.raw_root/'bop/models/models_info.json':
            raise ValueError('Only static CAD models_info metadata is allowed')
    def check(self,path,flags=os.O_RDONLY,native=False):
        if self.cad_metadata is not None and isinstance(path,(str,bytes,os.PathLike)) and Path(os.fsdecode(path)).resolve()==self.cad_metadata:
            if native or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):raise PermissionError('CAD metadata is read-only')
            self.counts['static_cad_metadata_reads']+=1;return
        super().check(path,flags,native)
    def snapshot(self):
        result=super().snapshot();result['counts']['static_cad_metadata_reads']=self.counts['static_cad_metadata_reads'];result['static_cad_metadata']=str(self.cad_metadata) if self.cad_metadata else None;return result
