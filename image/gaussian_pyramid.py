import torch
from torch import nn
import torch.nn.functional as nnf

import torchvision.transforms.functional as TF

# Note: Right now only handling factors of 2
def pyr_depth(depth, mode, kernel_size):
    stride = kernel_size

    if mode == "bilinear":
        new_depth = nnf.avg_pool2d(depth, kernel_size, stride)
    elif mode == "nearest_neighbor":
        new_depth = depth[:,:,0::stride,0::stride]
    elif mode == "max":
        new_depth = nnf.max_pool2d(depth, kernel_size)
    elif mode == "min":
        new_depth = -nnf.max_pool2d(-depth, kernel_size)
    elif mode == "masked_bilinear":
        mask = ~depth.isnan()
        depth_masked = torch.zeros_like(depth, device=depth.device)
        depth_masked[mask] = depth[mask]
        depth_sum = nnf.avg_pool2d(depth_masked, kernel_size, stride, divisor_override=1)
        mask_sum = nnf.avg_pool2d(mask.float(), kernel_size, stride, divisor_override=1)
        new_depth = torch.where(mask_sum > 0.0, depth_sum/mask_sum, torch.tensor(0.0, dtype=depth.dtype, device=depth.device))
    else:
        raise ValueError("pyr_depth mode: " + mode + " is not implemented.")

    return new_depth

def resize_depth(depth, mode, size):
    if mode == "bilinear":
        new_depth = TF.resize(depth, size, interpolation = TF.InterpolationMode.BILINEAR)  
    elif mode == "nearest_neighbor":
        new_depth = TF.resize(depth, size, interpolation = TF.InterpolationMode.NEAREST) 
    else:
        raise ValueError("resize_depth mode: " + mode + " is not implemented.")

    return new_depth


def resize_intrinsics(K, image_scale_factors):
    # T = torch.tensor([[image_scale_factors[1], 0, 0.5*image_scale_factors[1]-0.5],
    #                 [0, image_scale_factors[0], 0.5*image_scale_factors[0]-0.5],
    #                 [0, 0, 1]], device=K.device, dtype=K.dtype)
    T = torch.tensor([[image_scale_factors[1], 0, image_scale_factors[1]],
                      [0, image_scale_factors[0], image_scale_factors[0]],
                      [0, 0, 1]], device=K.device, dtype=K.dtype)
    K_new = torch.matmul(T,K)
    return K_new


class GaussianBlurModule(nn.Module):
    def __init__(self, channels, device, dtype):
        super(GaussianBlurModule, self).__init__()

        # Matches opencv documentation
        gaussian_kernel = (1.0/16.0) * torch.tensor([ [1.0, 2.0, 1.0], 
                                                  [2.0, 4.0, 2.0],
                                                  [1.0, 2.0, 1.0] ], requires_grad=False, device=device, dtype=dtype)
        self.gaussian_kernel = gaussian_kernel.repeat(channels,1,1,1)

    def forward(self, x):
        x_blur =  nn.functional.conv2d(nn.functional.pad(x, (1,1,1,1), mode='reflect'),
                                       self.gaussian_kernel, groups=x.shape[1])
        return x_blur


class ImagePyramidModule(nn.Module):
    def __init__(self, channels, start_level, end_level, device, dtype):
        super(ImagePyramidModule, self).__init__()

        self.blur_module = GaussianBlurModule(channels=channels, device=device, dtype=dtype)
        self.start_level = start_level
        self.end_level = end_level

    def forward(self, x):
        pyr = []
        x_level = x
        for i in range(self.end_level-1):
            if i >= self.start_level:
                pyr.insert(0, x_level)
            x_level = self.blur_module(x_level)[:,:,0::2,0::2]
        pyr.insert(0, x_level)
        return pyr

class DepthPyramidModule(nn.Module):
    def __init__(self, start_level, end_level, mode, device):
        super(DepthPyramidModule, self).__init__()

        self.start_level = start_level
        self.end_level = end_level
        self.mode = mode

    def forward(self, x):
        pyr = []
        x_level = x
        for i in range(self.end_level-1):
            if i >= self.start_level:
                pyr.insert(0, x_level)
            x_level = pyr_depth(x_level, self.mode, kernel_size=2)
        pyr.insert(0, x_level)
        return pyr

class IntrinsicsPyramidModule(nn.Module):
    def __init__(self, start_level, end_level, device):
        super(IntrinsicsPyramidModule, self).__init__()

        self.start_level = start_level
        self.end_level = end_level

    def forward(self, K_orig, image_scale_start):
        pyr = []
        for i in range(self.start_level, self.end_level):
            y_scale = image_scale_start[0]*pow(2.0, -i)
            x_scale = image_scale_start[1]*pow(2.0, -i)
            K_level = resize_intrinsics(K_orig, [y_scale, x_scale])
            pyr.insert(0, K_level)
        return pyr