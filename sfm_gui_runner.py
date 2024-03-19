import torch
import open3d as o3d
import open3d.visualization.gui as gui

from gui.sfm_gui import SfMWindow
from gui.odometery_gui import OdomWindow
import argparse
import sys
import numpy as np
import random
from pathlib import Path

def main(config_path, odom=False):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Open3D visualization setup
    app = gui.Application.instance
    app.initialize()

    if odom:
        viz_window = OdomWindow(config_path)
    else:
        viz_window = SfMWindow(config_path)
    app.run()

if __name__ == "__main__":
    # simple argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--odom', action='store_true', help='Run odometry visualiser', default=False)
    parser.add_argument('--config', type=str, help='Path to config file', default='configs/replica_sfm_example.yaml')

    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    main(config_path, args.odom)