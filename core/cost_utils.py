import torch


def split_by_mode(src_pixels, mode='colour'):
    if mode == 'colour':
        src_pixels_affine = src_pixels[:, :3]
        src_pixels_cosine = None
        src_kappa = None
    elif mode == 'colour_norm':
        src_pixels_affine, src_pixels_cosine = torch.split(src_pixels, [3, 3], dim=1)
        src_kappa = None
    elif mode == 'colour_norm_kappa':
        src_pixels_affine, src_pixels_cosine, src_kappa = torch.split(src_pixels, [3, 3, 1], dim=1)
    elif mode == 'norm_kappa':
        src_pixels_cosine, src_kappa = torch.split(src_pixels, [3, 1], dim=1)

        src_pixels_affine = None

    return src_pixels_affine, src_pixels_cosine, src_kappa

