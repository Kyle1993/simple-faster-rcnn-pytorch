from collections import namedtuple
from string import Template

import chainer.functions as F
import cupy,torch
import cupy as cp
import torch as t
from cupy.cuda import function
from torch.autograd import Function

from roi_cupy import kernel_backward,kernel_forward

Stream = namedtuple('Stream', ['ptr'])
@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)
CUDA_NUM_THREADS = 1024
def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K
   

class ROI(Function):
    """
    NOTE：only CUDA-compatible
    """
    def __init__(self,outh,outw,spatial_scale):
        self.forward_fn = load_kernel('roi_forward',kernel_forward)
        self.backward_fn = load_kernel('roi_backward',kernel_backward)
        self.outh,self.outw,self.spatial_scale = outh,outw,spatial_scale
        
    def forward(self,x,rois):
        self.in_size = B, C, H, W = x.size()
        N = rois.size(0)
        output = t.zeros(N, C, self.outh, self.outw).cuda()
        self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
        self.rois = rois
        args = [x.data_ptr(),rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale,C,H,W,
                self.outh,self.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                block=(CUDA_NUM_THREADS,1,1),
                grid=(GET_BLOCKS(top_data.numel()),1,1),
                stream=stream)
        return output

    def backward(self,grad_output):
        grad_input = t.zeros(self.in_size).cuda()
        stream=Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                N,spatial_scale,C,H,W,PH,PW,
                grad_input.numel()]
        self.backward_fn(args=args,
            block=(CUDA_NUM_THREADS,1,1),
            grid=(GET_BLOCKS(grad_input.numel()),1,1),
            stream = stream
            )
        return grad_input,None


class ROIPooling2D(t.nn.Module):

    def __init__(self, outh,outw,spatial_scale):
        super(ROIPooling2D, self).__init__()
        self.ROI = ROI(outh,outw,spatial_scale)

    def forward(self,x,rois):
        return self.ROI(x,rois)


def test_roi_module():

    ## fake data###
    B,N,C,H,W,PH,PW = 2,8,4,32,32,7,7

    bottom_data = t.randn(B,C,H,W).cuda()
    bottom_rois = t.randn(N,5)
    bottom_rois[:int(N/2),0]=0
    bottom_rois[int(N/2):,0]=1
    bottom_rois[:,1:] = (t.rand(N,4)*100).float()
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1./16
    outh,outw = PH,PW
    
    # pytorch version
    module = ROIPooling2D(outh,outw,spatial_scale)
    x = t.autograd.Variable(bottom_data,requires_grad=True)
    rois = t.autograd.Variable(bottom_rois)
    output = module(x,rois)
    output.sum().backward()
    grad_x = x.grad.cpu().data.numpy()

    def t2c(variable):
        npa = variable.data.cpu().numpy()
        return cp.array(npa)

    def test_eq(variable,array,info):
        cc=cp.asnumpy(array.data)
        neq = (cc!=variable.data.cpu().numpy())
        assert neq.sum()==0 ,'test failed: %s' %info

    # chainer version
    import chainer.functions as F
    x_cn = Variable(t2c(x))
    from chainer import Variable
    o_cn = F.roi_pooling_2d(x_cn, t2c(rois), outh, outw, spatial_scale)
    test_eq(output,o_cn,'forward')
    F.sum(o_cn).backward()
    test_eq(x.grad, x_cn.grad,'backward')
    print('test pass')
