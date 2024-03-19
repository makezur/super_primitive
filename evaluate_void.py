# script adapted from https://github.com/isl-org/VI-Depth/blob/main/evaluate.py
import os
import argparse
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

import yaml

from depth_completion.segment_based_completion import DepthCompletion
from depth_completion.fill_in_tools import fill_single_griddata, fill_depth
# import pipeline
import depth_completion.void as metrics
from tool.etc import to_np
import copy
from pathlib import Path
import random
from prettytable import PrettyTable
import cv2

seed = 144
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

# if name 
if __name__ == '__main__':
    # parse args 
    parser = argparse.ArgumentParser(description='Evaluate depth completion on VOID dataset')
    parser.add_argument('--config', type=str, default='config/depth_completion/void_dataset.yaml', help='path to config file')
    parser.add_argument('--dataset', type=str, default='/mnt/hdd/VOID/', help='path to the void dataset')
    parser.add_argument('--output', type=str, default='/mnt/hdd/void_refactor_run', help='path to config file')
    parser.add_argument('---save', type=bool, default=False, help='save results')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    # dataset_path = config['dataset']

    depth_completion = DepthCompletion(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0


    # get inputs
    dataset_path = Path(args.dataset)
    with open(dataset_path / 'test_image.txt') as f: 
        test_image_list = [line.rstrip() for line in f]
        
    avg_error_w_pred = metrics.ErrorMetricsAverager()
    avg_error_w_pred_valid = metrics.ErrorMetricsAverager()

    validity_rates = []

    save_result = False

    depth_save_path = None
    if save_result:
        depth_save_path = Path(args.output)
        depth_save_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(test_image_list))):

        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = read_image(input_image_fp)

        # intrinsics path
        K_path =  Path(input_image_fp).parent.parent / 'K.txt'
        K = np.loadtxt(K_path)
        # print(K)
        # sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        # sparse depth validity map
        validity_map_fp = input_image_fp.replace("image", "validity_map")
        validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        assert(np.all(np.unique(validity_map) == [0, 256]))
        validity_map[validity_map > 0] = 1

        # target (ground truth) depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth


        input_sparse_depth_t = torch.from_numpy(input_sparse_depth)
        input_image = (input_image * 255).astype(np.uint8)

        depths_zbuff, invalid_depth = depth_completion.depth_completion(input_image, K, input_sparse_depth_t)

        depths_zbuff = to_np(depths_zbuff)
        depths_zbuff_filled = fill_single_griddata(copy.copy(depths_zbuff), invalid_depth)

        if save_result:
            scene_name = Path(input_image_fp).parent.parent.name
            # print(depth_save_path)
            curr_save_path = depth_save_path / scene_name
            curr_save_path.mkdir(parents=True, exist_ok=True)

            name =  Path(input_image_fp).name + '_pred.npy'
            np.save(curr_save_path / name, depths_zbuff)
            np.save(curr_save_path /  (Path(input_image_fp).name + '_valid.npy'), 
                    invalid_depth)

        mask_joint = np.logical_and(depths_zbuff > 1e-6, mask)

        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = depths_zbuff_filled, 
            target = target_depth, 
            valid = mask,
        )         
        avg_error_w_pred.accumulate(error_w_pred)


        error_w_pred_partial = metrics.ErrorMetrics()
        error_w_pred_partial.compute(
            estimate = depths_zbuff, 
            target = target_depth, 
            valid = mask_joint,
        )  
        avg_error_w_pred_valid.accumulate(error_w_pred_partial)
    

        validity_rate = (depths_zbuff > 1e-6).sum() / (depths_zbuff.shape[0] *  depths_zbuff.shape[1])
        print('Validity rate?', validity_rate)
        validity_rates.append(validity_rate) 

    print("Averaging metrics for filled in depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()

    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "my"]

    summary_tb.add_row(["RMSE",  f"{avg_error_w_pred.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE",  f"{avg_error_w_pred.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel",  f"{avg_error_w_pred.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE",  f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel",  f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])

    print(summary_tb)



    print("Averaging metrics for parial depth over {} samples".format(
        avg_error_w_pred_valid.total_count
    ))
    avg_error_w_pred_valid.average()

    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "my"]

    summary_tb.add_row(["RMSE",  f"{avg_error_w_pred_valid.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE",  f"{avg_error_w_pred_valid.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel",  f"{avg_error_w_pred_valid.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE",  f"{avg_error_w_pred_valid.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_pred_valid.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel",  f"{avg_error_w_pred_valid.inv_absrel_avg:8.3f}"])

    print(summary_tb)


    print('Average validity rate', np.mean(validity_rates),
        'std', np.std(validity_rates),
            'min', np.min(validity_rates),
            'median', np.median(validity_rates))