import torch
import numpy as np
import numpy

import cupy
from cupyx.scipy import ndimage as ndi
import tool.point_utils as point_utils

from image.image_processing import ImageGradientModule
from tool.etc import to_np
import copy

def depth_discontinuity(logdepth, depth_validty, filter_size=3, threshold=0.1):
    torch.set_grad_enabled(False)
    # logdepth shape is [n_masks, h, w]
    # assume everything is torch
    depth = torch.exp(logdepth)
    depth[~depth_validty] = -1

    maxpooled = torch.nn.functional.max_pool2d(depth[:, None], filter_size, stride=1,
                                                padding=filter_size // 2)
    img_grad = ImageGradientModule(1, 
                                   logdepth.device,
                                   torch.float32)
    gx, gy = img_grad(maxpooled)
    grad = torch.sqrt(gx**2 + gy**2).squeeze(1)
    depth_d = (grad > threshold) * depth_validty
    torch.set_grad_enabled(True)
    return depth_d

def mask_by_depth_discontinuity(logdepth, depth_validity):
    torch.set_grad_enabled(False)
    disc = depth_discontinuity(logdepth, depth_validity)
    result = torch.logical_and(~disc, depth_validity)
    torch.set_grad_enabled(True)
    return result


def batch_label_connectivity():
    def _generate_binary_structure(rank, connectivity):
        if connectivity < 1:
            connectivity = 1
        if rank < 1:
            return numpy.array(True, dtype=bool)
        output = numpy.fabs(numpy.indices([3] * rank) - 1)
        output = numpy.add.reduce(output, 0)
        return output <= connectivity

    structure = _generate_binary_structure(2, 1)
    structure_pre = np.zeros_like(structure) > 0
    structure_post = np.zeros_like(structure) > 0
    structure = np.stack([structure_pre, structure, structure_post], axis=0)

    return structure


def connected_components_batch(masks):
    masks = cupy.asarray(masks)
    structure = batch_label_connectivity()

    cuda_labeled, num_labels = ndi.label(masks,
                                         structure=structure)
    
    return cupy.asnumpy(cuda_labeled), num_labels


def sample_pts_in_mask(masks):
    # [num_masks, H, W]
    # sample one point per mask within it

    pts_x, pts_y = [], []
    for mask_id in range(masks.shape[0]):
        x, y = torch.where(masks[mask_id])
        # select random
        idx = torch.randint(0, x.shape[0], (1,))[0]
        seed_x, seed_y = x[idx], y[idx]
        
        pts_x.append(seed_x)
        pts_y.append(seed_y)
    
    pts_x = torch.stack(pts_x, dim=0)
    pts_y = torch.stack(pts_y, dim=0)
    cat = torch.stack([pts_x, pts_y], dim=1)
    return cat

def remap_labels_to_arange(array):
    labels = np.unique(array)
    labels = np.sort(labels)
    arange = np.arange(len(labels))

    new_array = np.zeros_like(array)

    for old, new in zip(labels, arange):
        new_array[array == old] = new
    
    return new_array


def remap_cupylabel_outputs(connectivity):
    num_points = connectivity.shape[0]

    remaped = []
    for pt_id in range(num_points):
        mask_ids = connectivity[pt_id]
        mask_ids = remap_labels_to_arange(mask_ids)
        remaped.append(mask_ids)

    return np.stack(remaped, axis=0)

def post_process_kf(kf, connectivity, keep_ratio=1e-3):
    device = kf.keypoints.device
    num_points = kf.keypoints.shape[0]
    assert(kf.logdepth_perseg is not None)
    assert(num_points == connectivity.shape[0])
    H, W = kf.logdepth_perseg.shape[1:]
    all_masks = []
    all_logdepth = []
    all_keypoints = []

    for pt_id in range(num_points):
        # convert all masks to binary masks
        mask_ids = connectivity[pt_id]
        mask_ids = remap_labels_to_arange(mask_ids)
        mask_ids = torch.from_numpy(mask_ids).int().to(device)
        num_masks = mask_ids.max() + 1

        ids = torch.arange(num_masks, device=device)[:, None, None]
        bin_masks = (ids == mask_ids[None])
        ########

        segment_mask = kf.keypoint_regions[pt_id]
        torch.logical_and(segment_mask[None], bin_masks, out=bin_masks)
        part_sizes = bin_masks.sum(dim=(1, 2))

        part_sizes_ratio = ( part_sizes.float() / (H * W) ) > keep_ratio
        if part_sizes_ratio.sum() == 0:
            continue
        if part_sizes_ratio.sum() == 1:
            bin_masks = kf.keypoint_regions[pt_id][None]
            logdepth = kf.logdepth_perseg[pt_id][None]
            keypoints = kf.keypoints[pt_id][None]
        else:
            bin_masks = bin_masks[part_sizes_ratio]
            num_masks_keep = bin_masks.shape[0]
            # now shape is [num_masks_keep, H, W]
            logdepth = kf.logdepth_perseg[pt_id].expand(num_masks_keep, -1, -1)
            keypoints = sample_pts_in_mask(bin_masks)
            keypoints = point_utils.normalise_coordinates(keypoints, (H, W))
        
        all_masks.append(bin_masks)
        all_logdepth.append(logdepth)
        all_keypoints.append(keypoints)
    
    masks = torch.cat(all_masks, dim=0)
    logdepth = torch.cat(all_logdepth, dim=0)
    keypoints = torch.cat(all_keypoints, dim=0)
    return masks, logdepth, keypoints


def kf_fix_disconnected_regions(kf, filter_size=3, depth_threshold=0.1, 
                                area_keep_ratio=1e-3):
    discont = depth_discontinuity(kf.logdepth_perseg, 
                                  kf.keypoint_regions,
                                  filter_size=filter_size,
                                  threshold=depth_threshold)
    regions_with_discont = discont.sum(dim=(1,2))

    split = mask_by_depth_discontinuity(kf.logdepth_perseg, 
                                        kf.keypoint_regions)
    
    batch_con, _ = connected_components_batch(to_np(split))
    new_mask, new_logdepth, new_keypoints = post_process_kf(kf,
                                                            batch_con,
                                                            keep_ratio=area_keep_ratio)
    
    kf_new = copy.deepcopy(kf)
    kf_new.logdepth_perseg = new_logdepth
    kf_new.keypoint_regions = new_mask
    kf_new.keypoints = new_keypoints

    return kf_new
