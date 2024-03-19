import torch
import numpy as np

def img_to_np(img):
    if img.shape[0] == 1:
        img = img.squeeze(0)
    if img.shape[0] == 3:
        img = img.permute(1,2,0)
    img = img.detach().cpu().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    return img

def to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

# different convention for normalisation, works worse with segments + zero padding in image interpolation 
def normalise_coordinates_og(x_pixel, dims):
    inv = 1.0 / torch.as_tensor(dims, dtype=torch.float32, device=x_pixel.device)

    x_norm = 2 * x_pixel * inv + inv - 1
    return x_norm

def denormalise_coordinates_og(x_norm, dims):
    dims = torch.as_tensor(dims, dtype=torch.float32, device=x_norm.device)
    x_pixel = (x_norm) * dims / 2.0 + dims / 2.0 - 0.5
    return x_pixel.round().long()

def normalise_coordinates(x_pixel, dims):
    inv = 1.0 / (torch.as_tensor(dims, dtype=torch.float32, device=x_pixel.device) - 1)

    x_norm = 2 * x_pixel * inv - 1
    return x_norm

def denormalise_coordinates(x_norm, dims):
    dims = torch.as_tensor(dims, dtype=torch.float32, device=x_norm.device)
    x_pixel = 0.5 * (dims - 1) * ((x_norm) + 1 )
    return x_pixel.round().long()


def normalise_coordinates_np(x_pixel, dims):
    x_norm = normalise_coordinates(torch.from_numpy(x_pixel.copy()), dims)

    return to_np(x_norm)

def denormalise_coordinates_np(x_norm, dims):
    x_pixel = denormalise_coordinates(torch.from_numpy(x_norm.copy()), dims)

    return to_np(x_pixel)