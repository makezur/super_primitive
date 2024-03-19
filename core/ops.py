import torch



def transform_points_batch(points_3d, poses):
    # poses is (B, 4, 4)
    R_batch = poses[:, :3, :3]
    t_batch = poses[:, :3, 3]

    rotated = None

    if len(points_3d.shape) == 2:
        rotated = torch.einsum('bij, nj -> bni', R_batch, points_3d) 
    else:
        rotated = torch.einsum('bij, bnj -> bni', R_batch, points_3d) 
    
    return rotated + t_batch[:, None, :]

def project_points_batch(points_3d, K):
    # points_3d is (B, N, 3)
    # K is (B, 3, 3)
    eps = 1e-6
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    N_pts = points_3d.shape[1]
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    z_inv = torch.ones_like(z) * eps
    z_inv[torch.abs(z) > eps] = 1.0 / z[torch.abs(z) > eps] 

    # z_inv = torch.where(torch.abs(z).detach() < eps, torch.ones_like(z), 1.0 / z)
    u = x * fx[:, None] * z_inv + cx[:, None]
    v = y * fy[:, None] * z_inv + cy[:, None]

    return torch.stack([u, v], dim=-1)

def project_points(points_3d, K):
    return project_points_batch(points_3d[None], K[None])[0]

def transform_points(points_3d, pose):
    return transform_points_batch(points_3d[None], pose[None])[0]

def unproject_points_mat(points_2d, depth_2d, K):
    K_inv = torch.inverse(K)
    points_2d = points_2d.float()
    # transform to homogeneous coordinates
    points_2d = torch.cat([points_2d, torch.ones_like(points_2d[:, 0:1])], dim=1)

    points_3d = points_2d * depth_2d.reshape(-1, 1)
    points_3d = points_3d @ K_inv.T
    return points_3d


def estimate_depth_diff(points_3d, K, spatial_dim, mean=False):
    valid_depth = points_3d[..., 2].detach() > 1e-6
    
    # also mask_out points with zero depth

    # todo make sure this is correct
    with torch.no_grad():
        points_2d = project_points(points_3d, K).flip(-1).long()
    depth = points_3d[..., 2]

    # assert(torch.all(torch.isfinite(points_2d)))
    # create image tensor 
    image = torch.zeros((1, *spatial_dim), 
                         device=points_2d.device,
                         dtype=torch.float32)

    x = points_2d[..., 0]
    y = points_2d[..., 1]

    valid_depth = valid_depth * (x >= 0) * (x < spatial_dim[0]) * (y >= 0) * (y < spatial_dim[1])

    # print('x, y, depth', x.shape, y.shape, depth.shape)
    depth = depth[valid_depth]
    x = x[valid_depth]
    y = y[valid_depth]
    # print('x, y, depth', x.shape, y.shape, depth.shape)
    # print(image[:, x, y].shape)
    image_flat = image.reshape(-1)
    index = x * spatial_dim[1] + y
    if mean:
        image_flat.scatter_reduce_(0, index, depth, reduce='mean')
    else:
        image_flat.scatter_(0, index, depth)

    image = image_flat.reshape((1, *spatial_dim))
    # image[:, x, y] = depth

    return image, valid_depth