import torch 
from core.dense_optim import project_points, img_interp, infer_depth_seeds, unproject_points
from core.dense_optim import unproject_segments, expdepth, get_pixels
import tool.point_utils as point_utils
from core.normal_cost import transform_normals_batch
from core.dense_optim import calculate_residual

from core.ops import transform_points_batch, project_points_batch

from core.dense_optim import affine_compensation_batch_v2

def get_pixels_batch(image, points_3d, K, spatial_dim=None):
    # normalise points to [-1, 1] space
    # todo check if we need this or just zeros 
    valid_depth = points_3d[..., 2].detach() > 1e-6
    
    # also mask_out points with zero depth

    if len(points_3d.shape) == 2:
        points_2d = project_points(points_3d, K)
    else:
        points_2d = project_points_batch(points_3d, K)
    
    if not torch.isfinite(points_2d).all():
        assert(torch.all(torch.isfinite(points_2d)))

    if spatial_dim is None:
        spatial_dim = image.shape[1:]

    # this is stupid 
    swapped_xy = points_2d.flip(-1)
    points_norm = point_utils.normalise_coordinates(swapped_xy, 
                                                    spatial_dim)
    points_norm = points_norm.flip(-1)
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    if len(points_norm.shape) == 2:
        points_norm = points_norm.unsqueeze(0)

    image_samples, valid_mask = img_interp(image, points_norm)
    # filter out the points that are behind the camera
    valid_mask = torch.logical_and(valid_mask, valid_depth)
    
    return image_samples, valid_mask



def photomeric_cost_batch(src_keyframe, 
                          trg_images,
                          trg_Ks, 
                          src_keypoint_logdepth, 
                          poses,
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

    segment_depthes = src_points[..., 2]
    # debug stuff 
    result_stats = {}
    ###### 
    assert(torch.all(torch.isfinite(src_points)))
    src_in_trg_points = transform_points_batch(src_points, poses)

    # projected_trg_points = project_points(trg_points, trg_keyframe.K)
    if collect_stats > 1:
        src_keypoints_denorm = point_utils.denormalise_coordinates(src_keyframe.keypoints,
                                                                src_depth.shape[1:]).flip(-1)
        src_in_trg_keypoints = unproject_points(src_keypoints_denorm,
                                                expdepth(src_keypoint_logdepth), src_keyframe.K)
        src_in_trg_keypoints = transform_points_batch(src_in_trg_keypoints, poses)
        src_in_trg_keypoints_z = src_in_trg_keypoints[..., 2]
        _, src_trg_keypoints_valid_mask = get_pixels_batch(trg_images,
                                                        src_in_trg_keypoints, trg_Ks,
                                                    spatial_dim=spatial_size)
        
        src_in_trg_keypoints = project_points_batch(src_in_trg_keypoints, trg_Ks)

        result_stats = {
            'src_in_trg_keypoints': src_in_trg_keypoints,
            'src_in_trg_keypoints_z': src_in_trg_keypoints_z,
            'src_in_trg_keypoints_valid_mask': src_trg_keypoints_valid_mask
        }

    src_pixels, src_valid_mask = get_pixels(src_keyframe.image,
                                            src_points, src_keyframe.K,
                                            spatial_dim=spatial_size)
    
    src_pixels = transform_normals_batch(src_pixels, poses, mode=residual_mode)

    src_in_trg_pixels, trg_valid_mask = get_pixels_batch(trg_images,
                                                   src_in_trg_points, trg_Ks,
                                                   spatial_dim=spatial_size)

    full_mask =  trg_valid_mask[:, None].long() * src_valid_mask[:, None].long()

    if affine_comp is not None:
        affine_source, affine_target = affine_comp
        
        src_in_trg_pixels = affine_compensation_batch_v2(src_in_trg_pixels, affine_source,
                                                         affine_target)


    residual, residual_stats = calculate_residual(src_pixels, src_in_trg_pixels, full_mask, 
                                                cost_config,
                                                return_raw=collect_stats > 0,
                                                src_depthes=segment_depthes)
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


