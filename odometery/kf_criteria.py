import torch
from tool.etc import to_np
from scipy.spatial.transform import Rotation
import numpy as np


def translation_difference(pose_src, pose_target, depth):
    valid_depth = depth > 1e-6
    # infer scale from depth

    scale = torch.median(depth[valid_depth]) 

    translation_src = pose_src[:3, 3]
    translation_target = pose_target[:3, 3]
    difference = torch.linalg.norm(translation_src - translation_target)
    difference = difference / (scale + 1e-6)

    return difference, scale

def rotation_difference(pose_src, pose_target):
    pose_src = to_np(pose_src)
    pose_target = to_np(pose_target)

    pose_delta = np.linalg.inv(pose_src) @ pose_target

    rot_vec = Rotation.from_matrix(pose_delta[:3, :3]).as_rotvec()
    angular_difference = np.linalg.norm(rot_vec) * 180 / np.pi


    return angular_difference