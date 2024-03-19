from . import replica
from . import tum
from . import tum_undistort

def load_dataset(config):
    dataset = None
    if config['dataset']['type'] == 'replica':
        dataset = replica.ReplicaDataset(root_dir=config['dataset']['path'], 
                                         normal_dir=config['dataset']['normals_path'] if 'normals_path' in config['dataset'] else None)

    if config['dataset']['type'] == 'tum':
        dataset = tum.TUMDataset(config['dataset']['path'], 
                                 traj_file=config['dataset']['traj_file'])
        
    if config['dataset']['type'] == 'tum_undistort':
        dataset = tum_undistort.TUMDataset(config['dataset']['path'], 
                                           traj_file=config['dataset']['traj_file'])
        
    return dataset