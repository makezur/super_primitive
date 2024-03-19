from torch.utils.data import Dataset

import os
from pathlib import Path

import numpy as np
import cv2

from data import image_transforms
from torchvision import transforms

from frontend.normals.normals_inferer import load_gt_normals

def replica_K():
    w = 1024
    h = 768
    fx = 886.81
    fy = 886.81
    cx = 512.0
    cy = 384.0

    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    return K


class ReplicaDataset(Dataset):
    def __init__(self, root_dir, normal_dir):
        super().__init__()

        depth_scale = 1 / 1000
        max_depth = 10

        self.root_dir = Path(root_dir)
        self.normal_dir = None
        if normal_dir is not None:
            self.normal_dir = Path(normal_dir)

        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(depth_scale),
             image_transforms.DepthFilter(max_depth)])
        
    def __len__(self):
        return self.Twc.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.root_dir / f'rgb/rgb_{idx}.png'
        depth_path = self.root_dir / f'depth/depth_{idx}.png'

        image = cv2.imread(str(img_path)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(depth_path), -1).astype(np.float32)
        depth = self.depth_transform(depth)

        T = self.Twc[idx]

        normals, normals_mask = None, None

        if self.normal_dir is not None:
            normal_path = self.normal_dir / f'depth_{idx}_tblr_k3.png'
            normals, normals_mask = load_gt_normals(str(normal_path))
        
        return {
            'image': image,
            'depth': depth,
            'T': T,
            'normals': normals,
            'normals_mask': normals_mask,
            'intrinsics': replica_K()
        }
           