"""Exact integer serialization and padding adapters for pinned frozen Utonia.

Per-instance methods only: installed Utonia and checkpoint tensors are untouched.
The Hilbert transform follows the same forward-dimension Skilling operations as
Utonia's Apache-2.0 serialization/hilbert.py, using packed integers in one kernel.
"""
from types import MethodType
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N','D'])
def _code_kernel(G,B,O,N,D,TRANS:tl.constexpr,HILBERT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    x=tl.load(G+i*3+(1 if TRANS else 0),valid,0).to(tl.int64)
    y=tl.load(G+i*3+(0 if TRANS else 1),valid,0).to(tl.int64)
    z=tl.load(G+i*3+2,valid,0).to(tl.int64)
    if HILBERT:
        for bit in range(D-1,-1,-1):
            q=1<<bit;p=q-1
            x=x^tl.where((x&q)!=0,p,0)
            on=(y&q)!=0;t=tl.where(on,0,(x^y)&p)
            x=x^tl.where(on,p,t);y=y^t
            on=(z&q)!=0;t=tl.where(on,0,(x^z)&p)
            x=x^tl.where(on,p,t);z=z^t
    code=tl.full((BLOCK,),0,tl.int64)
    for bit in range(D):
        code=code|((x&(1<<bit))<<(2*bit+2))|((y&(1<<bit))<<(2*bit+1))|((z&(1<<bit))<<(2*bit))
    if HILBERT:
        for shift in tl.static_range(6):code=code^(code>>(1<<shift))
    batch=tl.load(B+i,valid,0).to(tl.int64)
    tl.store(O+i,code|(batch<<(D*3)),valid)


def encode(grid,batch,depth,order):
    assert order in ('z','z-trans','hilbert','hilbert-trans')
    output=torch.empty(len(grid),device=grid.device,dtype=torch.int64)
    _code_kernel[(triton.cdiv(len(grid),256),)](grid.contiguous(),batch,output,len(grid),depth,
        order.endswith('-trans'),order.startswith('hilbert'),256)
    return output


def unique_grid(grid):
    """Lexicographic unique for nonnegative, 16-bit coordinates, no roundoff."""
    keys=(grid[:,0].long()<<32)|(grid[:,1].long()<<16)|grid[:,2].long()
    keys,inverse,counts=torch.unique(keys,sorted=True,return_inverse=True,return_counts=True)
    values=torch.stack((keys>>32,(keys>>16)&65535,keys&65535),-1)
    return values,inverse,counts


def get_padding(self,point):
    if 'pad' not in point or 'unpad' not in point or 'cu_seqlens_key' not in point:
        # One host read per stage replaces scalar synchronization at every slice.
        ends=point.offset.tolist();counts=np.diff([0,*ends]);k=self.patch_size
        padded=np.where(counts>k,((counts+k-1)//k)*k,counts)
        starts=np.cumsum([0,*counts]);pstarts=np.cumsum([0,*padded])
        pad=np.arange(pstarts[-1],dtype=np.int64);unpad=np.arange(starts[-1],dtype=np.int64);cu=[]
        for i,count in enumerate(counts):
            start,end=int(starts[i]),int(starts[i+1]);ps,pe=int(pstarts[i]),int(pstarts[i+1])
            unpad[start:end]+=ps-start
            if count!=padded[i]:pad[ps+int(count):pe]-=k
            pad[ps:pe]-=ps-start;cu.extend(range(ps,pe,k))
        cu.append(int(pstarts[-1]))
        point['pad']=torch.from_numpy(pad).to(point.offset.device)
        point['unpad']=torch.from_numpy(unpad).to(point.offset.device)
        point['cu_seqlens_key']=torch.tensor(cu,device=point.offset.device,dtype=torch.int32)
    return point.pad,point.unpad,point.cu_seqlens_key


def install(encoder):
    from utonia.structure import Point
    from utonia.model import GridPooling,SerializedAttention
    from torch_scatter import segment_csr

    class FastPoint(Point):
        def serialization(self,order='z',depth=None,shuffle_orders=False):
            if shuffle_orders:raise ValueError('Frozen Utonia must keep fixed serialization order')
            self['order']=order
            if depth is None:depth=int(self.grid_coord.max()+1).bit_length()
            assert depth<=16 and depth*3+len(self.offset).bit_length()<=63
            self['serialized_depth']=depth
            code=torch.stack([encode(self.grid_coord,self.batch,depth,name) for name in order])
            indices=torch.argsort(code)
            inverse=torch.zeros_like(indices).scatter_(1,indices,torch.arange(code.shape[1],device=code.device).repeat(code.shape[0],1))
            self['serialized_code']=code;self['serialized_order']=indices;self['serialized_inverse']=inverse

    def pool(self,point):
        grid=torch.div(point.grid_coord,self.stride,rounding_mode='trunc').long()
        keys=(point.batch.long()<<48)|(grid[:,0]<<32)|(grid[:,1]<<16)|grid[:,2]
        keys,cluster,counts=torch.unique(keys,sorted=True,return_inverse=True,return_counts=True)
        grid=torch.stack(((keys>>32)&65535,(keys>>16)&65535,keys&65535),-1)
        _,indices=torch.sort(cluster);ptr=torch.cat([counts.new_zeros(1),counts.cumsum(0)])
        heads=indices[ptr[:-1]]
        data=dict(feat=segment_csr(self.proj(point.feat)[indices],ptr,reduce=self.reduce),
                  coord=segment_csr(point.coord[indices],ptr,reduce='mean'),grid_coord=grid,batch=point.batch[heads])
        for key in ('origin_coord','color'):
            if key in point:data[key]=segment_csr(point[key][indices],ptr,reduce='mean')
        for key in ('condition','context','name','split'):
            if key in point:data[key]=point[key]
        if 'grid_size' in point:data['grid_size']=point.grid_size*self.stride
        if self.traceable:data.update(pooling_inverse=cluster,pooling_parent=point,idx_ptr=ptr)
        order=point.order;point=FastPoint(data)
        if self.norm is not None:point=self.norm(point)
        if self.act is not None:point=self.act(point)
        point.serialization(order=order,shuffle_orders=self.shuffle_orders);point.sparsify()
        return point

    def forward(self,data_dict):
        point=FastPoint(data_dict);point=self.embedding(point)
        point.serialization(order=self.order,shuffle_orders=self.shuffle_orders);point.sparsify()
        point=self.enc(point)
        if not self.enc_mode:point=self.dec(point)
        return point

    encoder.forward=MethodType(forward,encoder)
    for module in encoder.modules():
        if isinstance(module,GridPooling):module.forward=MethodType(pool,module)
        if isinstance(module,SerializedAttention):module.get_padding_and_inverse=MethodType(get_padding,module)
