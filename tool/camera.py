import numpy as np
import torch


def instrinsic_scaled_K(K, scale): 
    K = K.copy()
    K[0,0] *= scale
    K[1,1] *= scale
    K[0,2] *= scale
    K[1,2] *= scale
    return K

def instrinsic_scaled_K_anisotropic(K, scale_H, scale_W): 
    if torch.is_tensor(K):
        K = K.detach().detach()
    else:
        K = K.copy()
    K[0,0] *= scale_W
    K[1,1] *= scale_H
    K[0,2] *= scale_W
    K[1,2] *= scale_H
    return K

def get_translation_norm(T):
    T = T.copy()
    t = T[:3,3]
    return np.linalg.norm(t)

def renorm_translation(T, t_norm, eps=1e-6):
    T = T.copy()
    t = T[:3,3]
    scaling_factor = t_norm / (np.linalg.norm(t) + eps)
    t = scaling_factor * t
    T[:3,3] = t
    return T, scaling_factor

def apply_scale(T, scaling_factor):
    T = T.copy()
    T[:3,3] *= scaling_factor
    return T
