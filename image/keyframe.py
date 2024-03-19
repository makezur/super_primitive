import torch
from torch import nn
import numpy as np

import tool.point_utils as point_utils
from tool.etc import image_tt
import image.gaussian_pyramid as gaussian_pyramid

import torchvision.transforms as TF

def infer_spatial_size(logdepth_perseg):
    perseg_mode = len(logdepth_perseg.shape) == 3
    if perseg_mode:
        spatial_size = logdepth_perseg.shape[1:]
    else:
        assert(len(logdepth_perseg.shape) == 2)
        spatial_size = logdepth_perseg.shape
    return spatial_size

class KeyFrame(nn.Module):
    def __init__(self, image, K, 
                 logdepth_perseg=None,
                 keypoints=None, keypoint_regions=None,
                 K_img=None, id=None):
        super().__init__()
        self.image = image
        self.K = K
        if K_img is None:
            K_img = K
        self.K_img = K_img

        self.id = id

        self.supporting = (logdepth_perseg is None) or (keypoints is None) or (keypoint_regions is None)

        self.logdepth_perseg = None
        self.keypoints = None
        self.keypoint_regions = None
        if not self.supporting:
            self.logdepth_perseg = logdepth_perseg
                
            # keypoints of size (N, 2)
            assert(keypoints.shape[0] == keypoint_regions.shape[0])
            self.keypoints = keypoints
            # mask of size (N, H, W)
            # here N is the number of keypoints
            self.keypoint_regions = keypoint_regions

    def geo_spatial_dim(self):
        return infer_spatial_size(self.get_logdepth())
    
    def get_logdepth(self):
        return self.logdepth_perseg

    def is_supporting(self):
        return self.supporting

    def normalised_keypoints(self):
        spatial_dim = self.image.shape[1:]
        swapped_xy = self.keypoints.flip(-1)
        return point_utils.normalise_coordinates(swapped_xy, 
                                                 spatial_dim)
    
    def num_segments(self):
        return self.keypoint_regions.shape[0]
    
    def __repr__(self):
        img_shape_string = f'{super().__repr__()} of shape {self.image.shape}'
        if not self.is_supporting():
            kpts = f'with {self.keypoints.shape[0]} keypoints'
        else:
            kpts = 'with no keypoints'
        
        return img_shape_string + '\n' + kpts


def keyframe_pyramid(keyframe, start_level, end_level, geo_down=False, drop_normals=False, grayscale=False):
    torch.set_grad_enabled(False)
    # todo implement me
    device = keyframe.image.device

    spatial_dim = keyframe.image.shape[1:]

    with_normals = keyframe.image.shape[0] > 3

    image_pyramid = gaussian_pyramid.ImagePyramidModule(1 if grayscale else 3, start_level, end_level, 
                                                        device=device,
                                                        dtype=keyframe.image.dtype)
    intriniscs_pyramid = gaussian_pyramid.IntrinsicsPyramidModule(start_level, 
                                                                end_level, 
                                                                device=device)
    depth_pyramid = None
    if not keyframe.is_supporting():
        depth_pyramid = gaussian_pyramid.DepthPyramidModule(start_level, end_level, 
                                                            mode='nearest_neighbor', 
                                                            device=device)
    normals_pyramid = None
    if with_normals:
        normals_pyramid = gaussian_pyramid.DepthPyramidModule(start_level, end_level, 
                                                            mode='nearest_neighbor', 
                                                            device=device)
    image_to_pyr = keyframe.image[:3][None]
    if grayscale:
        # to greyscale using torchvision
        grayscale_transform = TF.Grayscale(1) 
        image_to_pyr = grayscale_transform(image_to_pyr)
    image_pyr = image_pyramid(image_to_pyr)

    if not keyframe.is_supporting():
        depth_pyr = depth_pyramid(keyframe.logdepth_perseg.unsqueeze(1))
        mask_pyr = depth_pyramid(keyframe.keypoint_regions.int().unsqueeze(1))
    else:
        depth_pyr = [None] * len(image_pyr)
        mask_pyr = [None] * len(image_pyr)

    normals_pyr = [None] * len(image_pyr)
    if with_normals:
        normals_pyr = normals_pyramid(keyframe.image[3:][None])

    intr_pyr = intriniscs_pyramid(keyframe.K, [1.0, 1.0])

    keyframes = []
    for image, depth, mask, intr, norms in zip(image_pyr, depth_pyr, mask_pyr, intr_pyr, normals_pyr):
        image = image.squeeze(0)

        if norms is not None and not drop_normals:
            norms = norms.squeeze(0)
            image = torch.cat([image, norms], dim=0)
    
        if depth is not None:
            depth = depth.squeeze(1)
        if mask is not None:
            mask = mask.squeeze(1).bool()
        
        k = KeyFrame(image, 
                    K=intr if geo_down else keyframe.K.clone(),
                    logdepth_perseg=depth if geo_down else keyframe.logdepth_perseg,
                    keypoints=keyframe.keypoints, 
                    keypoint_regions=mask if geo_down else keyframe.keypoint_regions,
                    K_img=intr,
                    id=keyframe.id)
        
        # if geo_down:
        #     k = put_keypoints_back_kf(k)

        keyframes.append(k)
    torch.set_grad_enabled(True)
    return keyframes


def put_keypoints_back(keypoints, masks, logdepth_perseg=None):
    _, H, W = masks.shape
    keypoints_denorm = point_utils.denormalise_coordinates(keypoints, (H, W))
    num_valid_pix = masks.sum(dim=(1, 2))
    
    good_masks = num_valid_pix > 0
    keypoints_denorm = keypoints_denorm[good_masks]
    masks = masks[good_masks]
    if logdepth_perseg is not None:
        logdepth_perseg = logdepth_perseg[good_masks]

    for i in range(keypoints_denorm.shape[0]):
        x, y = keypoints_denorm[i]
        valid_x, valid_y = torch.where(masks[i])
        dist = torch.sqrt((valid_x - x)**2 + (valid_y - y)**2)
        idx = torch.argmin(dist)
        keypoints_denorm[i] = torch.tensor([valid_x[idx], valid_y[idx]])
    
    new_keypoints = point_utils.normalise_coordinates(keypoints_denorm, (H, W))
    if logdepth_perseg is not None:
        return new_keypoints, masks, logdepth_perseg
    return new_keypoints, masks

def put_keypoints_back_kf(keyframe):
    new_keypoints, new_regions, new_depth = put_keypoints_back(keyframe.keypoints,
                                                               keyframe.keypoint_regions, 
                                                               keyframe.logdepth_perseg)
    
    keyframe.keypoints = new_keypoints
    keyframe.keypoint_regions = new_regions
    keyframe.logdepth_perseg = new_depth
    return keyframe


