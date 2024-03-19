import torch
from torch import nn
import tool.point_utils as point_utils 
from core.cost_utils import split_by_mode
from core.normal_cost import transform_normals, transform_normals_batch

from core.ops import project_points


def infer_spatial_size(logdepth_perseg):
    perseg_mode = len(logdepth_perseg.shape) == 3
    if perseg_mode:
        spatial_size = logdepth_perseg.shape[1:]
    else:
        assert(len(logdepth_perseg.shape) == 2)
        spatial_size = logdepth_perseg.shape
    return spatial_size, perseg_mode

def unproject_points(points_2d, depth_2d, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # points_2d = points_2d.float()

    N_pts = points_2d.shape[0]
    assert(N_pts == depth_2d.shape[0])

    z = depth_2d.reshape(-1)
    x = (points_2d[:, 0].reshape(-1).float() - cx) * z / fx
    y = (points_2d[:, 1].reshape(-1).float() - cy) * z / fy
    # xy = (points_2d - c[None]) * z / f[None]

    return torch.stack([x, y, z], dim=1) 


def infer_depth_seeds(keypoint_logdepth, keypoints, keypoint_regions, logdepth_perseg):
    # keypoint_logdepth (N)
    # keypoints (N, 2)
    # keypoint_regions (N, H, W)
    num_pts = keypoints.shape[0]
    # print('keypoint_logdepth', keypoint_logdepth)
    assert(torch.isfinite(keypoint_logdepth).all())
    spatial_size, perseg_mode = infer_spatial_size(logdepth_perseg)

    if perseg_mode:
        assert(logdepth_perseg.shape[0] == num_pts)
        

    keypoints_denorm = point_utils.denormalise_coordinates(keypoints, 
                                                           spatial_size)
    x, y = keypoints_denorm[:, 0], keypoints_denorm[:, 1]
    x, y = x.long(), y.long()

    if perseg_mode:
        b = torch.arange(num_pts, device=x.device)

        unscaled_keypoints_depth = logdepth_perseg[b, x, y]
    else:
        unscaled_keypoints_depth = logdepth_perseg[x, y]

    # we're operating in log space, so we know log depth up to a shift
    depth_shifts = keypoint_logdepth - unscaled_keypoints_depth

    # now spread these shifts 
    # we could be smarter and store segments as crops along their bounding boxes 

    if perseg_mode:
        logdepth = logdepth_perseg
    else:
        logdepth = logdepth_perseg.unsqueeze(0).expand(num_pts, -1, -1)

    logdepth = logdepth + depth_shifts[:, None, None]
    # mask out regions that are not in the keypoint regions

    logdepth = logdepth * keypoint_regions
    assert(torch.all(torch.isfinite(logdepth)))

    return logdepth


def expdepth(logdepth):
    # return the jacobian later 
    
    return torch.exp(logdepth)


def unproject_segments(segment_depths,
                       segment_masks,
                       K, include_coords=False):
    # segment depths (N, H, W)
    # segment masks (N, H, W) is a bool mask with region validity 
    # K (3, 3) is the camera intrinsics matrix
    N, H, W = segment_depths.shape

    # this is for sanity checking
    num_pts_per_segment = torch.sum(segment_masks, dim=[1, 2])
    num_pts = num_pts_per_segment.sum()

    # infer x and y coordinates from the segment masks
    # and collect the depths 
    b, x, y = torch.where(segment_masks)

    depth_pts = segment_depths[b, x, y]

    assert(len(depth_pts) == num_pts)

    coords = torch.stack([y, x], dim=1)

    if include_coords:
        return unproject_points(coords, depth_pts, K), b, coords
    else:
        return unproject_points(coords, depth_pts, K), b


def transform_points(points_3d, pose):
    # jacobian wrt R is R * skew(points)
    R = pose[:3, :3]
    t = pose[:3, 3]

    return torch.matmul(points_3d, R.T) + t


# img is (B, C, H, W)
# sparse coords is (B, N, 2)
# output is (B, C, N)
def img_interp(img, coords_norm, mode="bilinear"):
    valid_mask = (torch.abs(coords_norm) <= 0.99)
    valid_mask = torch.all(valid_mask, dim=-1)

    x_samples = torch.unsqueeze(coords_norm, dim=1)

    sampled_vals = torch.nn.functional.grid_sample(img, x_samples, 
                                                   mode=mode,
                                                   padding_mode='zeros', align_corners=True)
    # Now convert to (B, 3*C, N)
    sampled_vals = torch.squeeze(sampled_vals, dim=2)

    return sampled_vals, valid_mask


def get_pixels(image, points_3d, K, spatial_dim=None, mode='bilinear'):
    # normalise points to [-1, 1] space
    # todo check if we need this or just zeros 
    valid_depth = points_3d[..., 2].detach() > 1e-7
    
    # also mask_out points with zero depth

    points_2d = project_points(points_3d, K)
    
    if spatial_dim is None:
        spatial_dim = image.shape[1:]

    points_norm = point_utils.normalise_coordinates(points_2d, 
                                                    (spatial_dim[1], spatial_dim[0]) )

    image_samples, valid_mask = img_interp(image[None], points_norm[None], mode=mode)
    # filter out the points that are behind the camera
    valid_mask = torch.logical_and(valid_mask, valid_depth)
    
    return image_samples, valid_mask

def unproject_kf_to_depths(kf, keypoint_logdepth):
    spatial_size = kf.geo_spatial_dim()

    src_segment_logdepths = infer_depth_seeds(keypoint_logdepth,
                                              kf.keypoints, 
                                              kf.keypoint_regions,
                                              kf.get_logdepth())
    
    src_depth = expdepth(src_segment_logdepths)
    
    return src_depth

def unproject_kf(kf, keypoint_logdepth, jacobian=False):
    spatial_size = kf.geo_spatial_dim()

    src_segment_logdepths = infer_depth_seeds(keypoint_logdepth,
                                              kf.keypoints, 
                                              kf.keypoint_regions,
                                              kf.get_logdepth())
    
    src_depth = expdepth(src_segment_logdepths)
    
    src_points, segm_id = unproject_segments(src_depth, 
                                             kf.keypoint_regions,
                                             kf.K)
    
    src_pixels, src_valid_mask = get_pixels(kf.image,
                                            src_points, kf.K,
                                            spatial_dim=spatial_size)
    
    result = {'src_pixels': src_pixels,
              'src_valid_mask': src_valid_mask,
              'src_pts': src_points,
              'segm_ids': segm_id,
              'spatial_size': spatial_size
              }
    return result

def affine_compensation_batch_v2(trg_pixels, 
                                 src_affine_comp, trg_affine_comp):
    rgb, etc = trg_pixels[:, :3], trg_pixels[:, 3:]
    if src_affine_comp is None:
        assert(trg_affine_comp is None)
        return trg_pixels
    
    if len(src_affine_comp.shape) == 1:
        src_affine_comp = src_affine_comp.unsqueeze(0)
    if len(trg_affine_comp.shape) == 1:
        trg_affine_comp = trg_affine_comp.unsqueeze(0)
    
    src_a, src_b = torch.split(src_affine_comp, [1, 1], dim=-1)
    src_a = src_a[:, None].expand(-1, 3, -1)
    src_b = src_b[:, None].expand(-1, 3, -1)

    trg_a, trg_b = torch.split(trg_affine_comp, [1, 1], dim=-1)
    trg_a = trg_a[:, None].expand(-1, 3, -1)
    trg_b = trg_b[:, None].expand(-1, 3, -1)
    
    a = trg_a - src_a
    b = trg_b - src_b
    rgb = torch.exp(-a) * rgb + b
    return torch.cat([rgb, etc], dim=1)


def calculate_residual(src_pixels, trg_pixels, validity_mask, cost_conifg, return_raw=False, src_depthes=None):
    mode = cost_conifg['mode']

    norm_weight = 0.0
    normal_loss_mode = None

    if mode != 'colour':
        normal_loss_mode = cost_conifg['normal_loss'] # 'lecrec'
        norm_weight = cost_conifg['normal_weight'] #0.1

    src_pixels_affine, src_pixels_cosine, src_kappa = split_by_mode(src_pixels, mode=mode)
    trg_pixels_affine, trg_pixels_cosine, trg_kappa = split_by_mode(trg_pixels, mode=mode)

    residual_affine = 0.0
    residual_cosine = 0.0

    residual_raw = None

    depth_reg_loss = 0.0

    if src_pixels_affine is not None and trg_pixels_affine is not None:
        residual_affine = (src_pixels_affine - trg_pixels_affine) * validity_mask
        if return_raw:
            residual_raw = residual_affine.detach().clone()
        
        residual_affine = torch.abs(residual_affine).mean(dim=[1, 2])


        # [1, 1, num_pts]
    median_depth = None

    stats = {'residual_raw': residual_raw,
             'median_depth': median_depth,}
    return residual_affine + norm_weight * residual_cosine + depth_reg_loss, stats



def photomeric_cost(src_keyframe, 
                    trg_keyframe, 
                    src_keypoint_logdepth, 
                    pose,
                    cost_config,
                    affine_comp=None):
    residual_mode = cost_config['mode']
    collect_stats = cost_config['collect_stats']

    # for each keypoint measurment, scale depth unscaled appropriately within each segment
    spatial_size = src_keyframe.geo_spatial_dim()

    src_segment_logdepths = infer_depth_seeds(src_keypoint_logdepth,
                                              src_keyframe.keypoints, 
                                              src_keyframe.keypoint_regions,
                                              src_keyframe.get_logdepth())
    
    
    src_depth = expdepth(src_segment_logdepths)
    
    src_points, segm_id = unproject_segments(src_depth, 
                                             src_keyframe.keypoint_regions,
                                             src_keyframe.K)

    # debug stuff 
    result_stats = {}
    if collect_stats > 1:
        src_keypoints_denorm = point_utils.denormalise_coordinates(src_keyframe.keypoints,
                                                                src_depth.shape[1:]).flip(-1)
        src_in_trg_keypoints = unproject_points(src_keypoints_denorm,
                                                expdepth(src_keypoint_logdepth), src_keyframe.K)
        src_in_trg_keypoints = transform_points(src_in_trg_keypoints, pose)
        src_in_trg_keypoints_z = src_in_trg_keypoints[:, 2]
        _, src_trg_keypoints_valid_mask = get_pixels(trg_keyframe.image,
                                                    src_in_trg_keypoints, trg_keyframe.K,
                                                    spatial_dim=spatial_size)
        
        src_in_trg_keypoints = project_points(src_in_trg_keypoints, trg_keyframe.K_img)

        result_stats = {
            'src_in_trg_keypoints': src_in_trg_keypoints,
            'src_in_trg_keypoints_z': src_in_trg_keypoints_z,
            'src_in_trg_keypoints_valid_mask': src_trg_keypoints_valid_mask
        }

    ###### 
    assert(torch.all(torch.isfinite(src_points)))
    src_in_trg_points = transform_points(src_points, pose)


    src_pixels, src_valid_mask = get_pixels(src_keyframe.image,
                                            src_points, src_keyframe.K,
                                            spatial_dim=spatial_size)
    
    src_pixels = transform_normals(src_pixels, pose, mode=residual_mode)

    assert(torch.all(torch.isfinite(src_in_trg_points)))
    src_in_trg_pixels, trg_valid_mask = get_pixels(trg_keyframe.image,
                                                   src_in_trg_points, trg_keyframe.K,
                                                   spatial_dim=spatial_size)


    full_mask =  trg_valid_mask[:, None].long() * src_valid_mask[:, None].long()


    if affine_comp is not None:
        affine_source, affine_target = affine_comp
        
        src_in_trg_pixels = affine_compensation_batch_v2(src_in_trg_pixels, affine_source,
                                                         affine_target)

    residual, residual_stats = calculate_residual(src_pixels, src_in_trg_pixels, full_mask, 
                                                cost_conifg=cost_config, 
                                                return_raw=collect_stats > 0)

    assert(torch.all(torch.isnan(src_points) == False))
    assert(torch.all(torch.isnan(src_pixels) == False))

    assert(torch.all(torch.isnan(residual) == False))
    
    result = {'residual': residual}

    if collect_stats > 0:
        result_info = {'segm_ids': segm_id,
                    'src_pixels': src_pixels,
                    'src_in_trg_pixels': src_in_trg_pixels,
                    'src_valid_mask': src_valid_mask,
                    'trg_valid_mask': trg_valid_mask,
                    'full_mask': full_mask,
                    'src_pts': src_points,
                    'src_in_trg_pts': src_in_trg_points,
                    }
        result.update(result_info)
        result.update(residual_stats)

        if collect_stats > 1:
            result.update(result_stats)

    return result

def photomeric_cost_precomputed(src_precomputed,
                                trg_keyframe, 
                                pose,
                                cost_config,
                                affine_comp=None):
    residual_mode = cost_config['mode']
    collect_stats = cost_config['collect_stats']

    src_points = src_precomputed['src_pts']
    src_pixels = src_precomputed['src_pixels']
    src_valid_mask = src_precomputed['src_valid_mask']
    spatial_size = src_precomputed['spatial_size']


    src_in_trg_points = transform_points(src_points, pose)

    src_pixels = transform_normals(src_pixels, pose, mode=residual_mode)


    src_in_trg_pixels, trg_valid_mask = get_pixels(trg_keyframe.image,
                                                   src_in_trg_points, trg_keyframe.K,
                                                   spatial_dim=spatial_size)


    full_mask =  trg_valid_mask[:, None].long() * src_valid_mask[:, None].long()

    if affine_comp is not None:
        affine_source, affine_target = affine_comp
        
        src_in_trg_pixels = affine_compensation_batch_v2(src_in_trg_pixels, affine_source,
                                                         affine_target)
        
    residual, _ = calculate_residual(src_pixels, src_in_trg_pixels, full_mask,
                                     cost_conifg=cost_config, return_raw=False)


    result = {'residual': residual}

    return result




