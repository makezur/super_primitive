from torch.utils.data import Dataset
import numpy as np
import trimesh
import cv2
import torch
from torchvision import transforms

class BGRtoRGB(object):
    """bgr format to rgb"""

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale


class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        return depth

class TUMDataset(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None):

        self.t_poses = None
        if traj_file is not None:
            with open(traj_file) as f:
                lines = (line for line in f if not line.startswith('#'))
                self.t_poses = np.loadtxt(lines, delimiter=' ')
        scale = 1 / 5000
        self.max_depth = 10
        
        rgb_transform = transforms.Compose(
            [BGRtoRGB()])
        depth_transform = transforms.Compose(
            [DepthScale(scale),
             DepthFilter(self.max_depth)])

        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform



        
        w = 640
        h = 480
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5
            # update intrinsic
        self.intrinsic = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]])

        self.associations_file = root_dir + "associations.txt"
        with open(self.associations_file) as f:
            timestamps, self.rgb_files, self.depth_files = zip(
                *[(float(line.rstrip().split()[0]),
                    line.rstrip().split()[1],
                    line.rstrip().split()[3]) for line in f])

            self.timestamps = np.array(timestamps)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = self.root_dir + self.depth_files[idx]
        rgb_file = self.root_dir + self.rgb_files[idx]

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.t_poses is not None:
            rgb_timestamp = self.timestamps[idx]
            timestamp_distance = np.abs(rgb_timestamp - self.t_poses[:, 0])
            gt_idx = timestamp_distance.argmin()
            quat = self.t_poses[gt_idx][4:]
            trans = self.t_poses[gt_idx][1:4]

            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans

        sample = {"image": image, "depth": depth, "T": T, "intrinsics": self.intrinsic}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample