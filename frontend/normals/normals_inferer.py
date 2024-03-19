import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

from frontend.normals.scannet_model import NNET as NNET_scannet

def predict_normals(model, image, resize=True, resize_ratio=0.5, resize_back=True):
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = image.astype(np.float32) / 255.0
    img = torch.from_numpy(img).to(0).permute(2, 0, 1).unsqueeze(0)
    img = normalize(img)

    if resize:
        orig_H, orig_W = img.size(2), img.size(3)
        if isinstance(resize_ratio, float):
            new_H, new_W = int(resize_ratio * (orig_H)), int(resize_ratio * (orig_W))
        else:
            new_H, new_W = resize_ratio
            new_H, new_W = int(new_H), int(new_W)
        # new_H, new_W = int(resize_ratio * (orig_H)), int(resize_ratio * (orig_W))
        # print(f'Resized image for normal inference, new_H: {new_H}, new_W: {new_W}')
        img = F.interpolate(img, size=(new_H, new_W), mode='bilinear')
    norm_out_list, _, _ = model(img, gt_norm_mask=None, mode='test')
    norm_out = norm_out_list[-1]

    if resize_back:
        norm_out = F.interpolate(norm_out, size=(orig_H, orig_W), mode='nearest')

    return norm_out

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def patched_argparse():
    args = dotdict({
        "NNET_architecture": "eesnu__B5__NF2048__BN",
        "NNET_output_dim": 4,
        "NNET_output_type": "G",
        "NNET_sampling_ratio": 0.4,
        "NNET_importance_ratio": 0.7,
    })
    return args


def setup_normals_predictor(config_path, verbose=False, scannet=True):
    args = patched_argparse()
    hparams = dotdict(vars(args))
    if verbose:
        print(hparams)
        
    # define model
    model = NNET_scannet(hparams).to(0)

    print('Setting up scannet pretrained normal predictor')
    ckpt_paths = [config_path]
    def load_checkpoint(fpath, model):
        ckpt = torch.load(fpath, map_location='cpu')['model']

        load_dict = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                k_ = k.replace('module.', '')
                load_dict[k_] = v
            else:
                load_dict[k] = v

        model.load_state_dict(load_dict)
        return model

    model = load_checkpoint(ckpt_paths[0], model)
    model.eval()

    return model


def load_gt_normals(normal_path):
    normal = cv2.cvtColor(cv2.imread(normal_path,  cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
    normal_mask = np.sum(normal, axis=2, keepdims=True) > 0
    normal = (normal.astype(np.float32)/255) * 2.0 - 1.0

    pred_norm = normal[None]

    return pred_norm, normal_mask 