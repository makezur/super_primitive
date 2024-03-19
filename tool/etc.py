
import torch
import numpy as np


def dict_cpu(d):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in d.items()}

def list_cpu(l):
    return [v.detach().cpu() for v in l]

def attrs_on_cpu(obj):
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in obj.__dict__.items()}

def to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_img(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    tensor = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return tensor 

def to_img_np(tensor):
    img = to_img(tensor)
    img = (img * 255).astype(np.uint8)
    return img

def from_np(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(array.copy())


def image_tt(image, device='cuda'):
    image_torch = (torch.from_numpy(image) / 255.).float().to(device)
    image_torch = image_torch.permute(2, 0, 1)
    return image_torch