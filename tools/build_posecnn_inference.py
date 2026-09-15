"""Build the released PoseCNN inference kernels in a private task directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from torch.utils.cpp_extension import load


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    changes={};paths=[]
    for name in ('hough_voting_kernel.cu','roi_pooling_kernel.cu','hard_label_kernel.cu'):
        source=a.source/'lib/layers'/name;target=a.out/name;text=source.read_text()
        if name=='hough_voting_kernel.cu':
            text=text.replace('#include <Eigen/Geometry> ','#include <thrust/execution_policy.h>')
        if name=='hard_label_kernel.cu':text=text.replace('bottom_prob.type()','bottom_prob.scalar_type()')
        target.write_text(text);paths.append(str(target))
        changes[name]=dict(source_sha256=sha(source),built_source_sha256=sha(target))
    bindings=a.out/'bindings.cpp';bindings.write_text('''#include <torch/extension.h>
#include <vector>
std::vector<at::Tensor> hough_voting_cuda_forward(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int,int,int,float,float,float);
std::vector<at::Tensor> roi_pool_cuda_forward(int,int,float,at::Tensor,at::Tensor);
std::vector<at::Tensor> hard_label_cuda_forward(float,float,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("hough_voting_forward",&hough_voting_cuda_forward);
 m.def("roi_pool_forward",&roi_pool_cuda_forward);
 m.def("hard_label_forward",&hard_label_cuda_forward);
}
''');paths.insert(0,str(bindings))
    module=load(name='lip_posecnn_inference_cuda',sources=paths,build_directory=str(a.out),verbose=True,
        extra_cflags=['-O2'],extra_cuda_cflags=['-O2'])
    receipt=dict(completed=True,upstream_commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
        files=changes,bindings_sha256=sha(bindings),module=str(Path(module.__file__).resolve()),module_sha256=sha(module.__file__),
        torch_cuda_arch_list=os.environ.get('TORCH_CUDA_ARCH_LIST'),
        scope='Original hough/ROI/hard-label CUDA math. Remove unused Eigen header, include Thrust execution policy and update AT_DISPATCH dtype API. Private extension only; no global CUDA or driver modifications. Training/SDF/FP kernels are not provided.')
    (a.out/'build_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
