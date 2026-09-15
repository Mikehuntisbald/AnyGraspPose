"""Fetch only the s0 member of the author's public Box checkpoint archive."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib
import requests


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--proxy');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);session=requests.Session();session.trust_env=False
    if a.proxy:session.proxies={'http':a.proxy,'https':a.proxy}
    shared='https://utdallas.box.com/s/g0qnbs615kcizcqvys6pn96dr251m0lk'
    url='https://utdallas.box.com/index.php?rm=box_download_shared_file&shared_name=g0qnbs615kcizcqvys6pn96dr251m0lk&file_id=f_1072157515293'
    session.get(shared,timeout=30).raise_for_status();size=4185722388
    def fetch(start,end):
        for attempt in range(4):
            try:
                response=session.get(url,headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'},timeout=(30,90))
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                    raise RuntimeError('Unexpected range response '+str((response.status_code,response.headers.get('Content-Range'))))
                if len(response.content)!=end-start+1:raise RuntimeError('Truncated range')
                return response.content
            except (requests.RequestException,RuntimeError):
                if attempt==3:raise
                time.sleep(2)
    end=fetch(size-65536,size-1);i=end.rfind(b'PK\x05\x06');assert i>=0
    eocd=struct.unpack_from('<4s4H2LH',end,i);offset=eocd[6]-(size-len(end));entry=None
    while end[offset:offset+4]==b'PK\x01\x02':
        c=struct.unpack_from('<4s6H3L5H2L',end,offset);name=end[offset+46:offset+46+c[10]].decode()
        if name=='checkpoints/dex_ycb_s0/vgg16_dex_ycb_epoch_16.checkpoint.pth':
            entry=dict(name=name,method=c[4],crc32=c[7],compressed=c[8],uncompressed=c[9],offset=c[16])
        offset+=46+sum(c[10:13])
    assert entry and entry['method']==8
    header=fetch(entry['offset'],entry['offset']+1023);h=struct.unpack_from('<4s5H3L2H',header)
    assert h[0]==b'PK\x03\x04' and header[30:30+h[9]].decode()==entry['name']
    start=entry['offset']+30+h[9]+h[10];compressed=a.out/'s0.deflate.part'
    position=compressed.stat().st_size if compressed.exists() else 0
    assert position<=entry['compressed']
    with compressed.open('ab') as f:
        while position<entry['compressed']:
            count=min(16*1024**2,entry['compressed']-position);block=fetch(start+position,start+position+count-1)
            f.write(block);f.flush();position+=len(block);print(json.dumps(dict(compressed_bytes=position,total=entry['compressed'])),flush=True)
    destination=a.out/'vgg16_dex_ycb_s0_epoch16.pth';temporary=destination.with_suffix('.pth.tmp')
    decoder=zlib.decompressobj(-15);crc=0;written=0;digest=hashlib.sha256()
    with compressed.open('rb') as source,temporary.open('wb') as target:
        while block:=source.read(1024**2):
            data=decoder.decompress(block);target.write(data);crc=zlib.crc32(data,crc);written+=len(data);digest.update(data)
        data=decoder.flush();target.write(data);crc=zlib.crc32(data,crc);written+=len(data);digest.update(data)
    assert decoder.eof and not decoder.unused_data and crc==entry['crc32'] and written==entry['uncompressed']
    temporary.replace(destination)
    receipt=dict(completed=True,source_page='https://github.com/IRVLUTD/posecnn-pytorch',shared_archive=shared,
        archive_bytes=size,archive_reported_sha1='8686164b3a5da6979a026937ae471f21f9ef94bf',
        archive_sha1_independently_verified=False,central_directory_tail_sha256=hashlib.sha256(end).hexdigest(),
        member=entry,member_crc_verified=True,checkpoint=str(destination.resolve()),checkpoint_sha256=digest.hexdigest(),
        scope='Public author-lab mirror of PoseCNN checkpoints. HTTP ranges retrieve the s0 member; full archive SHA1 is not claimed verified.')
    (a.out/'download_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
