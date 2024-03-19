from core import dense_optim

import torch
from tool.etc import to_np
from core import ops

def estimate_depth_kf_native(kf, kf_logdepth, pose=None, mean=False):
    with torch.no_grad():
        points = dense_optim.unproject_kf(kf, kf_logdepth)
        points = points['src_pts']

        if pose is None:
            pose = torch.eye(4, device=kf_logdepth.device)

        points = dense_optim.transform_points(points, pose)
        depth_render, validity = ops.estimate_depth_diff(points, 
                                                        kf.K, 
                                                        kf.geo_spatial_dim(),
                                                        mean=mean)
    
    return depth_render[0]
