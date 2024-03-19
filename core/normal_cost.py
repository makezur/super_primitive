import torch
from core.cost_utils import split_by_mode


def transform_normals(src_pixels, pose, mode='colour'):
    if mode == 'colour':
        return src_pixels
    
    return transform_normals_batch(src_pixels, pose[None], mode)

def transform_normals_batch(src_pixels, poses, mode='colour'):
    assert(src_pixels.shape[0] == 1)
    if mode == 'colour':
        return src_pixels
    
    batch_size = poses.shape[0]
    new_src_pixels = src_pixels.expand(batch_size, -1, -1)

    _, src_pixels_cosine, _ = split_by_mode(src_pixels, mode=mode)
    new_src_pixels, _, new_src_kappa = split_by_mode(new_src_pixels, mode=mode)
    
    R = poses[:, :3, :3].detach()
    new_normals = torch.einsum('bij, bjn -> bin', R, src_pixels_cosine)
    # print('new normals', new_normals.shape)
    to_cat = [new_src_pixels, new_normals]

    if new_src_kappa is not None:
        to_cat.append(new_src_kappa)

    return torch.cat(to_cat, dim=1)

