"""Bounded recovery of known s0 training sequences from completed archive chunks.
Does not change the downloaded archive. Dataset outputs stay under project cache.
Known positions 0..3 are explicitly listed in the pinned official toolkit README.
"""
import io,json,pathlib,tarfile,time
archive=pathlib.Path('/mnt/why/DexYCB/dex-ycb-20210415.tar.gz.part');markers=archive.parent/'.download';i=0
while (markers/f'{i:05d}.json').exists():i+=1
limit=i*33554432
out=pathlib.Path('cache/raw_train_subset');out.mkdir(parents=True,exist_ok=True)
known=['20200709_141754','20200709_141841','20200709_141931','20200709_142022']
class Prefix(io.RawIOBase):
 def __init__(self):self.f=archive.open('rb');self.remaining=limit
 def read(self,n=-1):
  if self.remaining<=0:return b''
  n=min(self.remaining,n if n>=0 else self.remaining);b=self.f.read(n);self.remaining-=len(b);return b
seen=[];saved=0;reader=Prefix()
try:
 with tarfile.open(fileobj=reader,mode='r|gz') as tar:
  for m in tar:
   parts=pathlib.PurePosixPath(m.name).parts
   if len(parts)==2 and m.isdir():seen.append(parts[1]);print('SEQUENCE',parts[1],flush=True)
   if len(parts)<3 or parts[0]!='20200709-subject-01' or parts[1] not in known or not m.isfile():continue
   if '..' in parts or m.name.startswith('/'):raise RuntimeError('Unsafe archive member')
   # Keep all RGB/depth/object labels and sequence metadata; read no hand arrays.
   if parts[-1]!='meta.yml' and not parts[-1].startswith(('color_','aligned_depth_to_color_','labels_')):continue
   p=out.joinpath(*parts);p.parent.mkdir(parents=True,exist_ok=True);src=tar.extractfile(m)
   data=src.read()
   if len(data)!=m.size:raise EOFError('Incomplete archive member')
   p.write_bytes(data);saved+=1
except (tarfile.ReadError,EOFError) as e:print('Reached bounded prefix:',str(e),flush=True)
report=dict(source_archive=str(archive),verified_prefix_bytes=limit,known_train_sequence_positions={s:j for j,s in enumerate(known)},source='NVlabs/dex-ycb-toolkit README at 64551b001d360ad83bc383157a559ec248fb9100',seen=seen,saved_files=saved,scope='partial official s0 train only; not full split')
(out/'subset_provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
