from pathlib import Path
import pickle
import torch

def dump_kf(kf_save_path,
            kf, kf_pose, kf_logdepth, kf_affine, kf_timestamp):
        save_path = Path(kf_save_path)
        save_file = save_path / f'kf_{kf_timestamp}.pkl'

        dump_dict = {'kf': kf,
                     'kf_logdepth': kf_logdepth,
                     'kf_pose': kf_pose,
                     'kf_affine': kf_affine,
                     'kf_timestamp': kf_timestamp}
        
        with open(str(save_file), 'wb') as f:
            pickle.dump(dump_dict, f)

def load_kf(kf_save_path, kf_timestamp):
    save_path = Path(kf_save_path)
    save_file = save_path / f'kf_{kf_timestamp}.pkl'

    with open(str(save_file), 'rb') as f:
        dump_dict = pickle.load(f)
    
    return dump_dict['kf'], dump_dict['kf_pose'], dump_dict['kf_logdepth'],  dump_dict['kf_affine'], dump_dict['kf_timestamp']
