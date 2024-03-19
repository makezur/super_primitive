import torch
import sys

import copy
from scipy import ndimage as nd
import yaml

import odometery.depth_init as depth_init
from frontend import process_frame

from core.dense_optim import unproject_kf_to_depths
from tool.etc import to_img_np, to_np

import tool.point_utils as point_utils


# def fill_depth(depth, invalid_mask):
#     ind = nd.distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
#     return depth[tuple(ind)]

def render_depth_avg(depths):
    invalid_depth = depths.max(dim=0)[0] < 1e-6
    depths[depths < 1e-6] = 0.0

    num_valid_depths_per_pixel = ((depths > 1e-6).sum(dim=0)  + 1e-6)
    depths = depths.sum(dim=0) / num_valid_depths_per_pixel
    return depths, invalid_depth


def infer_depth(front_processor, image, keypoints, K, partial_depth, rerun=False):
    orig_config = copy.deepcopy(front_processor.config)
    if rerun:
        # patch config to fallback into segmenting out larger regions
        front_processor.config['sam_params']['nms'] = False
        front_processor.config['sam_params']['select_smallest'] = False

    kf = front_processor.process_to_kf(image, K, keypoints=keypoints)

    if rerun:
        # revert to original config
        front_processor.config = orig_config

    partial_depth = partial_depth.to(kf.image.device)
    keypoints_logdepth, visible_seg = depth_init.segment_based_depth_reinit(partial_depth.clone().detach(),
                                                                            kf, mode='median', return_info=True)
    

    depths = unproject_kf_to_depths(kf, keypoints_logdepth)
    depths[kf.keypoint_regions == 0] = -1
    depths = depths[visible_seg]
    # remove unseeded regions


    depths, invalid_depth = render_depth_avg(depths)

    return depths, invalid_depth


class DepthCompletion():
    def __init__(self, config_path):
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.front_processor = process_frame.setup_new_front_processor(self.config)

    def depth_completion(self, image, K, partial_depth):
        
        x, y = torch.where(partial_depth > 1e-6)
        keypoints = torch.stack([x, y], dim=1).float()

        H, W = partial_depth.shape

        keypoints = point_utils.normalise_coordinates(keypoints, (H, W))


        keypoints = keypoints.to('cuda:0')
        # print('num keypoints', keypoints.shape)

        depths, invalid_depth = infer_depth(self.front_processor, image, keypoints, K, partial_depth)

        invalid_ratio = invalid_depth.sum().float() / (invalid_depth.shape[0] *  invalid_depth.shape[1])
        if invalid_ratio > 0.15:
            print('high invalid depth ration, reruning with larger masks')
            depths_new, invalid_depth_new = infer_depth(self.front_processor, 
                                                        image, keypoints, K, partial_depth,
                                                        rerun=True)
            

            depths[invalid_depth] = depths_new[invalid_depth]
            invalid_depth = torch.logical_and(invalid_depth, invalid_depth_new)
        depths = to_np(depths)
        invalid_depth = to_np(invalid_depth)

        return depths, invalid_depth
        