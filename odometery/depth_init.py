import torch
import tool.point_utils as point_utils

import numpy as np
import copy

from tool.etc import to_np
import core.dense_optim as dense_optim

def segment_based_depth_reinit(estimated_depth, kf, mode='mean', return_info=False):
    # make a more robust version of segment_based_reinit
    assert(mode == 'mean' or mode == 'median')
    device = kf.logdepth_perseg.device
    torch.set_grad_enabled(False)
    eps = 1e-6

    num_keypoints = kf.keypoints.shape[0]
    _, H, W = kf.logdepth_perseg.shape
    keypoints = point_utils.denormalise_coordinates(kf.keypoints, (H, W))
    b = torch.arange(num_keypoints, device=kf.logdepth_perseg.device)
    # i.e each keypoint is initialised with the median of the shifts 
    if isinstance(estimated_depth, np.ndarray): 
        estimated_depth = torch.from_numpy(estimated_depth).to(device)
    # estimated depth is of shape (H, W)
    invalid_estimate = estimated_depth < eps
    valid_estimate = ~invalid_estimate
    # do logdepth where valid
    estimated_depth[invalid_estimate] = eps
    estimated_logdepth = torch.log(estimated_depth)

    depth_shifts = estimated_logdepth[None] - kf.logdepth_perseg 
    # estimated depth shifts for each point in segment 
    # of shape (num_keypoints, H, W)
    valid_regions = kf.keypoint_regions * valid_estimate[None]
    depth_shifts = depth_shifts * valid_regions
    
    num_valid = (valid_regions).sum((1, 2))
    visible_segments = num_valid > 0

    depth_shifts_mean = torch.zeros((num_keypoints), device=device)

    if mode == 'mean':
        depth_shifts_mean[visible_segments] = depth_shifts[visible_segments].sum((1, 2)) / num_valid[visible_segments]
        depth_shifts_mean[visible_segments] += kf.logdepth_perseg[b, keypoints[:, 0], keypoints[:, 1]][visible_segments]
        depth_shifts_mean[~visible_segments] = torch.median(depth_shifts_mean[visible_segments])
    else:
        depth_shifts_valid = []
        visible_depth_shifts = depth_shifts[visible_segments]
        visible_validity = valid_regions[visible_segments]

        for pt_id in range(visible_depth_shifts.shape[0]):
            current_shifts = visible_depth_shifts[pt_id]
            current_validty = visible_validity[pt_id]
            # mask out invalid shifts
            current_shifts = current_shifts[current_validty]
            depth_shifts_valid.append(torch.median(current_shifts))
        depth_shifts_valid = torch.stack(depth_shifts_valid, dim=0)
        depth_shifts_mean[visible_segments] = depth_shifts_valid 
        depth_shifts_mean[visible_segments] += kf.logdepth_perseg[b, keypoints[:, 0], keypoints[:, 1]][visible_segments]
        depth_shifts_mean[~visible_segments] = torch.median(depth_shifts_mean[visible_segments])
    
    keypoints_logdepth = depth_shifts_mean
    torch.set_grad_enabled(False)
    if return_info:
        return keypoints_logdepth, visible_segments
    else:
        return keypoints_logdepth
