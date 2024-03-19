import torch
import numpy as np

from tool.camera import instrinsic_scaled_K
from submodules.batched_normal_integration import normal_integration_batch_cupy

def run_tiled_normal_integration(normals, intrinsics,
                                 mask,
                                 down_scale=1,
                                 cg_max_iter=1000,
                                 cg_tol=1e-3):
    K = instrinsic_scaled_K(intrinsics, 1 / down_scale)
    normals = normals[::down_scale, ::down_scale]
    mask = mask[:, ::down_scale, ::down_scale]

    normals = normals.detach().clone().contiguous()
    mask = mask.detach().clone().contiguous()

    integrated_depth = normal_integration_batch_cupy(normals,
                                                     mask,
                                                     K=K,
                                                     cg_max_iter=cg_max_iter,
                                                     cg_tol=cg_tol)

    expanded = torch.zeros_like(mask, dtype=torch.float32)
    expanded[mask > 0] = integrated_depth

    return expanded
