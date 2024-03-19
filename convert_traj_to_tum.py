import yaml

import pickle


from pathlib import Path

import data
import tool.pose_utils as pose_utils
from lie.lie_algebra import pose_to_tq


def write_tum_format(tum_timestamps, pred_poses, gt_poses, out_dir):
    # write in tum format
    # timestamp tx ty tz qx qy qz qw
    path = Path(out_dir)
    # make sure the directory exists
    path.mkdir(parents=True, exist_ok=True)
    output_file = path / 'converted_tum_traj.txt'
    output_file_gt = path / 'converted_gt_tum_traj.txt'

    with open(output_file, 'w') as f:
        for timestamp, pose in zip(tum_timestamps, pred_poses):
            f.write(f'{timestamp} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n')

    with open(output_file_gt, 'w') as f:
        for timestamp, pose in zip(tum_timestamps, gt_poses):
            f.write(f'{timestamp} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n')
    return 


def convert_to_tum(root_path):
    # root_path = Path(traj_path).parent.parent
    root_path = Path(root_path)
    traj_path = root_path / 'traj' / 'kf_traj_final.pkl'

    with open(traj_path, 'rb') as f:
        kf_traj = pickle.load(f)
    config_path = root_path / 'config.yaml'
    # # load conifg

    config = yaml.load(open(str(config_path), 'r'), Loader=yaml.FullLoader)
    dataset = data.load_dataset(config)

    poses, timestamps =  pose_utils.get_sorted_by_timestamp(kf_traj, return_ids=True)
    poses = [pose_to_tq(pose) for pose in poses]
    poses_gt = [pose_to_tq(dataset[int(ts)]['T']) for ts in timestamps]

    tum_timestamps = [dataset.timestamps[int(timestamp)] for timestamp in timestamps]

    print(f'writing tum format to {root_path}')
    write_tum_format(tum_timestamps, poses, poses_gt, str(root_path))
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()
    convert_to_tum(args.root)