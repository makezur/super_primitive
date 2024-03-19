import torch
import lietorch

import numpy as np
from scipy.spatial.transform import Rotation

import torch.nn.functional as F

### taken from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def renormalise_se3(matricies: torch.Tensor) -> torch.Tensor:
    # matrices either (4, 4) or (B, 4, 4)
    # m = matricies.detach().clone()
    R = matricies[..., :3, :3]
    R = quaternion_to_matrix(_matrix_to_quaternion_t(R))
    matricies[..., :3, :3] = R
    return matricies


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _matrix_to_quaternion_t(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def matrix_to_q_torch(matrix):
    # wrapper to move the real part to the end
    lie_qs = _matrix_to_quaternion_t(matrix)

    a, b = torch.split(lie_qs, [1, 3], dim=-1)
    lie_qs = torch.cat([b, a], dim=-1)
    return lie_qs

def torch_pose_to_tq(pose):
    if len(pose.shape) == 2:
        q = matrix_to_q_torch(pose[:3, :3])
        tq = torch.cat([pose[:3, 3], q], dim=0)
    elif len(pose.shape) == 3:
        q = matrix_to_q_torch(pose[:, :3, :3])
        tq = torch.cat([pose[:, :3, 3], q], dim=1)
        
    return tq

# Numpy 
# pose is (B,4,4) or (4,4)
# output is Bx7  (tx, ty, tz, qx, qy, qz, qw)
def pose_to_tq(pose):
    if len(pose.shape) == 2:
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        tq = np.concatenate([pose[:3, 3], q], axis=0)
    elif len(pose.shape) == 3:
        q = Rotation.from_matrix(pose[:, :3, :3]).as_quat()
        tq = np.concatenate([pose[:, :3, 3], q], axis=1)
        
    return tq

# Numpy
# tq is (B,7) or (7,)
def tq_to_pose(tq):

    if len(tq.shape) == 1:
        t = tq[:3] # tx, ty, tz
        q = tq[3:] # qx, qy, qz, qw
        T = np.empty((4,4))
        T[:3,:3] = Rotation.from_quat(q).as_matrix()
        T[:3,3] = t
        T[3,:3] = 0.0
        T[3,3] = 1.0

    elif len(tq.shape) == 2:
        t = tq[:,:3] # tx, ty, tz
        q = tq[:,3:] # qx, qy, qz, qw
        T = np.empty((tq.shape[0],4,4))
        T[:,:3,:3] = Rotation.from_quat(q).as_matrix()
        T[:,:3,3] = t
        T[:,3,:3] = 0.0
        T[:,3,3] = 1.0

    return T


def se3_exp(delta):
    delta_T_lietorch = torch.cat((delta[:,3:], delta[:,:3]), dim=1)
    T_lietorch = lietorch.SE3.exp(delta_T_lietorch).matrix()
    # T = SE3_expmap(delta.squeeze(0)).unsqueeze(0)
    return T_lietorch


def batch_se3(poses, delta_T):
    delta_T_lietorch = torch.cat((delta_T[:,3:], delta_T[:,:3]), dim=1)
    T_lietorch = lietorch.SE3.exp(delta_T_lietorch).matrix()
    poses_new = torch.matmul(poses, T_lietorch)
    return poses_new

# T is (..., 4, 4)
def invertSE3(T):
    T_inv = torch.empty_like(T)
    T_inv[...,:3,:3] = torch.transpose(T[...,:3,:3], dim0=-2, dim1=-1)
    T_inv[...,:3,3:4] = -torch.matmul(T_inv[...,:3,:3], T[...,:3,3:4])
    T_inv[...,3,:3] = 0.0
    T_inv[...,3,3] = 1.0
    return T_inv

def normalizeSE3_inplace(T):
    R = T[...,:3,:3]
    U, S, Vh = torch.linalg.svd(R)
    T[...,:3,:3] = torch.matmul(U, Vh)


def SO3_expmap(w):
    device = w.device
    dtype = w.dtype

    theta2 = torch.sum(torch.square(w))
    theta = torch.sqrt(theta2)
    sin_theta = torch.sin(theta)
    s2 = torch.sin(0.5*theta)
    one_minus_cos = 2.0*s2*s2

    W = torch.tensor([[0.0, -w[2], w[1], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]], device=device, dtype=dtype)
    K = W/theta
    KK = torch.matmul(K, K)

    R = torch.eye(3,device=device,dtype=dtype) + sin_theta*K + one_minus_cos*KK

    return R

def SO3_logmap(R, eps=1e-6):
    trace_R = R[...,0,0] + R[...,1,1] + R[...,2,2]
    tr_3 = trace_R - 3.0
    theta = torch.acos(0.5*(trace_R-1))
    # print(trace_R-3.0, theta)
    mag = torch.where(tr_3 < -eps, theta/(2.0*torch.sin(theta)), 0.5 - tr_3/12.0 + tr_3*tr_3/60.0)
    tmp_v = torch.stack((R[:,2,1]-R[:,1,2], R[:,0,2]-R[:,2,0], R[:,1,0]-R[:,0,1]), dim=1)
    w = mag * tmp_v
    return w

# P is (..., 3)
# Px is (..., 3, 3)
def skew_symmetric(P):
    size = list(P.shape)
    size.append(3)
    Px = torch.zeros(size, device=P.device, dtype=P.dtype)
    Px[...,0,1] = -P[...,2]
    Px[...,0,2] = P[...,1]
    Px[...,1,0] = P[...,2]
    Px[...,1,2] = -P[...,0]
    Px[...,2,0] = -P[...,1]
    Px[...,2,1] = P[...,0]
    return Px

def SE3_logmap(T, eps=1e-6):
    w = SO3_logmap(T[:,:3,:3])
    theta = torch.linalg.norm(w,dim=(1))
    theta = torch.clamp(theta, min=eps)
    w_norm = w/theta
    tan = torch.tan(0.5*theta)

    t = T[:,:3,3]
    wnorm_x_t = torch.cross(w_norm,t)
    V_inv_t = t - (0.5 * t) * wnorm_x_t + (1.0 - theta/(2.0*tan)) * torch.cross(w_norm, wnorm_x_t)
    xi = torch.cat((w, V_inv_t),dim=-1)

    return xi