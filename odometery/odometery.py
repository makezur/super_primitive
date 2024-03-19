import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import numpy as np


import yaml
import copy
import random
import shutil
from pathlib import Path
import pickle
import queue
import time
    

import core.dense_optim as dense_optim
import core.dense_optim_batch as dense_optim_batch 
import core.depth_render

import lietorch
import lie.lie_algebra as lie_algebra


import image.keyframe as keyframe 
import frontend.process_frame as process_frame
import tool.point_utils as point_utils
from lie.lie_algebra import invertSE3

import odometery.depth_init as depth_init
from odometery.utils import load_kf, dump_kf

import data 

from tool.etc import attrs_on_cpu, list_cpu, dict_cpu, from_np, to_np


from lie.lietorch_utils import lietorch_detach, lietorch_new_param, print_pose, zero_out_lietorch_tensor, mat_to_lie


from odometery.kf_criteria import translation_difference, rotation_difference    
from tool.etc import to_img_np, to_img 

def depth_median_loss(kf_logdepths):
    loss = 0.0

    for kf_logdepth in kf_logdepths:
        median_logdepth = torch.median(kf_logdepth)
        loss += torch.abs(kf_logdepth - median_logdepth).mean()

    return loss

class ParamsSupportingKF:
    def __init__(self, pose, timestamp, affine=None):
        self.pose = pose
        self.timestamp = timestamp
        self.affine = affine

    def update_params(self, new_pose, new_affine):
        self.pose = new_pose
        self.affine = new_affine

    def clonedetach(self):
        detached_affine = None
        if self.affine is not None:
            detached_affine = self.affine.detach().clone()
        self.pose = self.pose.detach().clone()
        self.affine = detached_affine

        return self
    
    # get print representation
    def __repr__(self):
        return f'Supporting KF params, timestamp: {self.timestamp}'

class SupportingKF:
    def __init__(self, kf, timestamp):
        self.kf = kf
        self.timestamp = timestamp

    # get print representation
    def __repr__(self):
        return f'Supporting KF, timestamp: {self.timestamp}'

class Odometery(mp.Process):
    def __init__(self, config_path, waitev):
        super().__init__()

        self.waitev = waitev
        # self.is_init = False

        self.opt_pose = True 

        self.config_path = config_path
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        self.window_size = self.config['window_size']
        self.mono_init = False
        if 'mono_init' in self.config['aligment']:
            self.mono_init = self.config['aligment']['mono_init']

        self.dump_kfs = False

        if 'dump_kfs' in self.config:
            self.dump_kfs = self.config['dump_kfs']

        self.save_every = -1
        if 'save_every' in self.config:
            self.save_every = self.config['save_every']
        self.affine_compensation = self.config['aligment']['affine_compensation']


        self.full_connectivity = False

        if 'full_connectivity' in self.config['aligment']:
            self.full_connectivity = self.config['aligment']['full_connectivity']

    def id_to_timestamp(self, frame_id):
        return str(frame_id).zfill(6)

    def init_keyframe(self, frame, pose, timestamp, affine=None):
        eps = 1e-6
        torch.set_grad_enabled(False)
        image = frame['image']
        K = frame['intrinsics']

        kf = self.front_processor.process_to_kf(image, K)
        H, W = kf.geo_spatial_dim()

        device = kf.keypoints.device

        if len(self.kfs) < 2:
            if self.mono_init:
                self.init_scale = 1.0
                keypoints_logdepth = self.init_scale * torch.ones_like(kf.keypoints[:, 0])
                keypoints_logdepth = torch.log(keypoints_logdepth)
            else:
                print('Using GT depth to Initialise!!')
                gt_depth = frame['depth']
                gt_depth = from_np(gt_depth).to(device)
                keypoints = point_utils.denormalise_coordinates(kf.keypoints, 
                                                                frame['depth'].shape)

                keypoints_logdepth = gt_depth[keypoints[:, 0], 
                                            keypoints[:, 1]]
                
                if (keypoints_logdepth < 1e-6).sum() > 0:
                    print('invalid GT depth, using smart init')
                    # resize gt depth to kf size
                    _, H_kf, W_kf = kf.logdepth_perseg.shape
                    gt_depth = TF.resize(from_np(frame['depth'])[None, None], (H_kf, W_kf), interpolation=TF.InterpolationMode.NEAREST)
                    gt_depth = to_np(gt_depth[0][0])
                    keypoints_logdepth = depth_init.segment_based_depth_reinit(gt_depth, kf, mode='median')
                else:
                    keypoints_logdepth = torch.log(keypoints_logdepth)
                del gt_depth
                self.initialised = True
        else:
            # here we want to initialise keypoints' depthes via the previous keyframe depth reprojection
            # we render the segments into the depth map of the new keyframe 
            estimated_depth = self.estimate_depth_latest_kf(pose)

            keypoints_logdepth = depth_init.segment_based_depth_reinit(estimated_depth, kf, mode='median')
        torch.set_grad_enabled(True)

        pose = pose.detach().clone()
        aff = None
        if self.affine_compensation:
            aff = affine.detach().clone()

        assert(torch.all(torch.isnan(keypoints_logdepth) == False))

        self.add_kf(kf=kf,
                    pose=pose, 
                    keypoints_logdepth=keypoints_logdepth,
                    timestamp=timestamp, 
                    aff=aff)
        
        # dump kf

        if self.window_size is not None and len(self.kfs) > self.window_size:
            if self.dump_kfs:
                print(f'Saving kf {self.kf_timestamps[0]} to {self.kf_save_path}')
                dump_kf(self.kf_save_path,
                        kf=self.kfs[0],
                        kf_pose=self.kf_poses[0],
                        kf_logdepth=self.kf_logdepths[0],
                        kf_affine=self.kf_affines[0] if self.affine_compensation else None,
                        kf_timestamp=self.kf_timestamps[0])
            
            self.pop_kf(0)

        return kf, pose, keypoints_logdepth, timestamp

    def init_supporting_frame(self, frame):
        current_T = self.current_track
        current_aff = None
        if self.affine_compensation:
            current_aff = self.current_aff.detach().clone()

        supp_f = self.front_processor.process_to_supp_kf(frame['image'], 
                                                         frame['intrinsics'],
                                                         device='cuda:0')
        
        return supp_f, current_T, current_aff

    def init_system(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.initialised = False
        self.paused = True
        if 'paused' in self.config:
            self.paused = self.config['paused']

        if self.opt_pose:
            def pose_to_mat(pose):
                if isinstance(pose, lietorch.LieGroupParameter):
                    pose = pose.retr()
                if pose.shape[0] > 1:
                    return pose.matrix()
                return pose.matrix()[0]
            self.pose_to_mat = lambda x: pose_to_mat(x)
        else:
            self.pose_to_mat = lambda x: x

        self.opt_supporting = True
        if 'opt_supporting' in self.config['aligment']:
            self.opt_supporting = self.config['aligment']['opt_supporting']

        self.reset_kfs()
        self.reset_running_supp_kfs()
        self.reset_new_supp_kfs()
        self.reset_tracked_poses()

        self.global_kf_trajectory = {}
        self.global_kf_scale = {}
        self.global_track_trajectory = {}

        self.current_aff = None

        self.renormalise_last_logdepth = False
        self.dataset = data.load_dataset(self.config)
        self.front_processor = process_frame.setup_new_front_processor(self.config)

        if self.save_every > 0:
            self.init_dirs()

        self.mapping_scheduled = False
         
    def raw_init(self):
        self.init_system()

        self.start_id = 000
        if 'start_id' in self.config['dataset']:
            self.start_id = int(self.config['dataset']['start_id'])
        self.current_frame_id = self.start_id

        self.current_aff = None 
        if self.affine_compensation:
            self.current_aff = torch.zeros((2,), device='cuda:0', dtype=torch.float32)
            
        start_timestamp = str(self.start_id).zfill(6)
        start_pose = torch.from_numpy(self.dataset[self.start_id]['T']).to('cuda:0').float()
        # start_pose = torch.eye(4).to('cuda:0').float()

        self.init_keyframe(self.dataset[self.start_id],
                           start_pose, 
                           start_timestamp,
                           self.current_aff)
        self.update_track_pose('init')

        self.global_kf_trajectory[start_timestamp] = start_pose.detach().cpu()
        self.global_kf_scale[start_timestamp] = 1.0
        
        cpu_kfs = [attrs_on_cpu(self.kfs[-1])]
        timestamps = [self.kf_timestamps[-1]]
        cpu_residuals = [dict_cpu(dense_optim.unproject_kf(self.kfs[-1], self.kf_logdepths[-1]))]              
        self.viz_queue.push(('init',
                    cpu_kfs,
                    timestamps,
                    list_cpu([self.kf_poses[-1]]),
                    cpu_residuals,
                    None,
                    ), block=True)
        return   

    def estimate_depth_latest_kf(self, pose):
        # estimate depth from the last keyframe
        pose_inv = torch.linalg.inv(pose)
        delta_pose = pose_inv @ self.kf_poses[-1] 
        return core.depth_render.estimate_depth_kf_native(self.kfs[-1], self.kf_logdepths[-1], delta_pose)
    
    def setup_tracking_opt(self, curr_aff=None):
        delta = lietorch_new_param(lietorch.SE3.Identity(1).to('cuda:0'))
        # here we only optmise for the current frame pose againts the latests keyframe
        track_params = [{'params': delta, 'lr': self.config['aligment']['track']['lr']}]
        if self.affine_compensation:
            assert(curr_aff is not None)
            curr_aff = nn.Parameter(curr_aff.detach().clone())
            aff_params = [{'params': curr_aff, 'lr': 5e-3}]
            track_params += aff_params

        optim = torch.optim.Adam(track_params, lr=5e-3)

        return delta, curr_aff, optim
    
    def apply_motion_prior(self, supp_T):
        if len(self.tracked_frames) < 2:
            return supp_T.detach().clone()
        prev_track = self.tracked_poses[-2]
        delta = self.current_track @ invertSE3(prev_track)
        delta = delta.detach().clone()

        return (delta @ supp_T).detach().clone()
    
    def track_frame(self, frame, timestamp):
        self.motion_prior = False

        cost_params = copy.deepcopy(self.config['aligment']['cost_params'])
        cost_params['mode'] = 'colour'
        cost_params['collect_stats'] = 2

        start_time = time.time()

        supp_kf, supp_T, curr_aff = self.init_supporting_frame(frame)

        if self.motion_prior:
            supp_T = self.apply_motion_prior(supp_T)


        with torch.no_grad():
            prev_kf = self.kfs[-1]
            prev_pose = self.kf_poses[-1].detach().clone()
            prev_logdepth = self.kf_logdepths[-1].detach().clone()
            prev_aff = None
            if self.affine_compensation:
                prev_aff = self.kf_affines[-1].detach().clone()

        delta, affine_comp, optim = self.setup_tracking_opt(curr_aff)


        with torch.no_grad():
            supp_kf_pyr = keyframe.keyframe_pyramid(supp_kf, self.config['aligment']['track']['pyramid_min'], 
                                                             self.config['aligment']['track']['pyramid_max'],
                                                            geo_down=False)
            
            prev_kf_pyr = keyframe.keyframe_pyramid(prev_kf, self.config['aligment']['track']['pyramid_min'], 
                                                             self.config['aligment']['track']['pyramid_max'],
                                                            geo_down=False)
            
        num_pyramid_levels = len(supp_kf_pyr)


        prev_loss = torch.inf
        stop_tol = 1e-8
        converged = False

        with torch.no_grad():
            kf_precomputed = []
            for pyr_level in range(num_pyramid_levels):
                kf_precomputed.append(dense_optim.unproject_kf(prev_kf_pyr[pyr_level],
                                                               prev_logdepth))


        num_iters = self.config['aligment']['track']['steps']


        for pyr_level in range(num_pyramid_levels):
            if converged:
                break

            for iter in range(num_iters[pyr_level]):
                if converged:
                    break

                with torch.no_grad():
                    src_pose = prev_pose
                    trg_pose = supp_T
                delta_pose = self.pose_to_mat(delta)

                residuals = dense_optim.photomeric_cost_precomputed(kf_precomputed[pyr_level],
                                                                    supp_kf_pyr[pyr_level],
                                                                    pose=delta_pose @ torch.linalg.inv(trg_pose) @ src_pose,
                                                                    affine_comp=(prev_aff, affine_comp) if self.affine_compensation else None,
                                                                    cost_config=cost_params)

                loss = torch.mean(residuals['residual'])
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

                # handle poses 
                with torch.no_grad():
                    supp_T = supp_T @ torch.linalg.inv(self.pose_to_mat(delta))
                # set delta to zero
                delta = zero_out_lietorch_tensor(delta)

                del residuals
        
        supp_T = lie_algebra.renormalise_se3(supp_T).detach().clone()
        # print('tracked affine', affine_comp)
        # send final
        cpu_kfs = [attrs_on_cpu(prev_kf_pyr[-1]), attrs_on_cpu(supp_kf_pyr[-1])]
        timestamps = [self.kf_timestamps[-1], timestamp]
        with torch.no_grad():
            pose_to_send = supp_T
        pose_to_send = list_cpu([pose_to_send])
        # cpu_residuals = [dict_cpu(residuals)]
        with torch.no_grad():
            self.viz_queue.push(('tracking',
                                cpu_kfs,
                                timestamps,
                                pose_to_send,
                                None,
                                timestamps,
                                ), block=False)
    
        stop_time = time.time()

        torch.cuda.synchronize()
        print(f'tracking time f{stop_time - start_time:.3} sec, tracking loss = {loss.item():.3}', flush=True)


        self.current_track = supp_T.detach().clone()
        if self.affine_compensation:
            self.current_aff = affine_comp.detach().clone()
        self.current_ts = timestamp

        self.tracked_frames.append(supp_kf)
        self.tracked_poses.append(self.current_track.detach().clone())
        self.tracked_timestamps.append(timestamp)
        if self.affine_compensation:
            self.tracked_affines.append(self.current_aff.detach().clone())

        self.global_track_trajectory[timestamp] = to_np(self.current_track)

        track_result = {'kf': supp_kf,
                        'pose': self.current_track.detach().clone(),
                        'affine': self.current_aff.detach().clone() if self.affine_compensation else None,
                        'ts': timestamp}
        return track_result
    

    def generate_connectivity_batch(self, mode):
        full_connectivity = False

        num_kfs = len(self.kfs)
        num_supp_kfs = len(self.curr_supp_kfs)

        connectivity = {}

        if full_connectivity :
            for src_id in range(num_kfs):
                # all other kfs are trg expcept iself
                trg_ids = list(range(num_kfs))
                trg_ids.remove(src_id)
                connectivity[src_id] = trg_ids
        else:
            # connect only neighbouring kfs
            for src_id in range(num_kfs):
                trg_ids = []
                if mode == 'supp' and src_id != num_kfs - 1:
                    continue

                if src_id > 0:
                    trg_ids.append(src_id - 1)
                if src_id < num_kfs - 1:
                    trg_ids.append(src_id + 1)

                connectivity[src_id] = trg_ids

        return connectivity
    
    def get_supp_kf_poses_pairs(self):
        latest_supp_kfs, latets_opt_kfs = self.running_supp_to_newformat()

        assert(len(self.supp_kfs_class[-1]) == 0)
        assert(len(self.supp_kfs_opt[-1]) == 0)

        kfs = [[] for _ in range(len(self.kfs))]
        poses = [[] for _ in range(len(self.kfs))]
        affines = [[] for _ in range(len(self.kfs))]
        timestamps = [[] for _ in range(len(self.kfs))]

        if not self.initialised:
            return {'kfs': kfs,
            'poses': poses,
            'affines': affines,
            'timestamps': timestamps}

        for kf_id in range(len(self.kfs)):
            if kf_id == len(self.kfs) - 1:
                current_params = latets_opt_kfs
                current_kfs = latest_supp_kfs
            else:
                current_params = self.supp_kfs_opt[kf_id]
                current_kfs = self.supp_kfs_class[kf_id]
            
            for kf, param in zip(current_kfs, current_params):
                detached = param.clonedetach()

                assert(kf.timestamp == detached.timestamp)

                kfs[kf_id].append(kf.kf)
                poses[kf_id].append(detached.pose)
                affines[kf_id].append(detached.affine)
                timestamps[kf_id].append(kf.timestamp)

        result = {'kfs': kfs,
                  'poses': poses,
                  'affines': affines,
                  'timestamps': timestamps}
    
        return result

    def setup_supporting_opt(self, mode, device='cuda:0'):
        latest_supp_kfs, latets_opt_kfs = self.running_supp_to_newformat()

        assert(len(self.supp_kfs_class[-1]) == 0)
        assert(len(self.supp_kfs_opt[-1]) == 0)

        delta_poses = [[] for _ in range(len(self.kfs))]
        delta_poses_opt = []

        affines = [[] for _ in range(len(self.kfs))]
        affines_opt = []

        if not self.initialised:
            return {'delta_supporting': delta_poses,
                    'delta_supporting_opt': delta_poses_opt,

                    'affine_supporting': affines,
                    'affine_supporting_opt': affines_opt}
        
        for kf_id in range(len(self.kfs)):
            if kf_id == len(self.kfs) - 1:
                current_params = latets_opt_kfs
            else:
                current_params = self.supp_kfs_opt[kf_id]
            
            for param in current_params:
                detached = param.clonedetach()
                
                delta = lietorch.SE3.Identity(1).to(device)
                affine = detached.affine

                if self.opt_supporting and mode != 'supp':
                    delta = lietorch_new_param(delta)
                    if self.affine_compensation:
                        affine = nn.Parameter(affine)
                
                delta_poses[kf_id].append(delta)
                if self.affine_compensation:
                    affines[kf_id].append(affine)
                
                if self.opt_supporting and mode != 'supp':
                    delta_poses_opt.append(delta)
                    if self.affine_compensation:
                        affines_opt.append(affine)
        
        
        results = {'delta_supporting': delta_poses,
                   'delta_supporting_opt': delta_poses_opt,

                   'affine_supporting': affines,
                   'affine_supporting_opt': affines_opt}
        return results
    
    def setup_mapping_opt(self, mode, device='cuda:0'):
        detach_depth = True
        # KF poses
        if mode == 'init' and self.mono_init:
            lr_pose = 1e-2
        else:
            lr_pose = 1e-4

        lr_logdepth = 1e-2      
        lr_affine = 1e-5

        if mode == 'supp':
            delta_poses_opt = []
            delta_poses = [lietorch.SE3.Identity(1).to(device) for _ in self.kf_poses]
        else:
            delta_poses_opt =  [lietorch_new_param(lietorch.SE3.Identity(1).to(device)) for _ in self.kf_poses[1:]]
            delta_poses = [lietorch.SE3.Identity(1).to(device)] + delta_poses_opt 

        if len(self.kfs) == self.window_size and detach_depth:
            logdepths_to_optim = [logdepth.detach().clone() for logdepth in self.kf_logdepths[1:]]

            logdepths_to_optim = [nn.Parameter(logdepth) for logdepth in logdepths_to_optim]
            logdepths = [self.kf_logdepths[0].detach().clone()] + logdepths_to_optim
        else:
            logdepths_to_optim = [logdepth.detach().clone() for logdepth in self.kf_logdepths]

            logdepths_to_optim = [nn.Parameter(logdepth) for logdepth in logdepths_to_optim]
            logdepths = logdepths_to_optim

        if mode == 'supp':
            del logdepths, logdepths_to_optim
            logdepths = [logdepth.detach().clone() for logdepth in self.kf_logdepths]
            logdepths[-1] = nn.Parameter(logdepths[-1])
            logdepths_to_optim = [logdepths[-1]]

        supp_opt_data = self.setup_supporting_opt(mode, device=device)
        supp_params = []

        if self.opt_supporting:
            supp_params = [{'params': supp_opt_data['delta_supporting_opt'], 'lr': lr_pose},
                           {'params': supp_opt_data['affine_supporting_opt'], 'lr': lr_affine}]

            if mode == 'supp':
                supp_params = []


        aff_params = []
        affine_optim = [None] * len(self.kfs)
        if self.affine_compensation:
            affine_optim = [self.kf_affines[0].detach().clone()] + [nn.Parameter(aff.detach().clone()) for aff in self.kf_affines[1:]]
            aff_params = [{'params': affine_optim[1:],  'lr': lr_affine}]

        if mode == 'supp':
            aff_params = []

        optim_params =  [{'params': logdepths_to_optim,   'lr': lr_logdepth},
                         {'params': delta_poses_opt,      'lr': lr_pose},
                            ] + aff_params

        if self.opt_supporting:
            optim_params += supp_params
        
        optim = torch.optim.Adam(optim_params, lr=1e-3)

        opt_result = {'optimiser': optim,
                      'optim_params': optim_params,
                      'logdepths': logdepths,
                      'delta_poses': delta_poses,
                      'affine': affine_optim,
                      'supp_affine': supp_opt_data['affine_supporting'],
                      'supp_delta_poses': supp_opt_data['delta_supporting']}
    
        return opt_result

    
    def collect_target_frames(self, mode, supp_kfs, supp_kf_ts, stack=True):
        kfs = self.kfs
        kf_connectivity = self.generate_connectivity_batch(mode)
        
        torch.set_grad_enabled(False)
        trg_images_precomputed = [[] for _ in range(len(kfs))]
        trg_images_precomputed_ts = [[] for _ in range(len(kfs))]

        for src_kf_id, trg_kf_ids in kf_connectivity.items():
            # collect kf images according to connectifity
            for trg_kf_id in trg_kf_ids:
                next_kf = kfs[trg_kf_id]
                trg_images_precomputed[src_kf_id].append(next_kf.image)
                trg_images_precomputed_ts[src_kf_id].append(self.kf_timestamps[trg_kf_id])

            # use supporting frames for the next kf too
            supp_src_ids = [src_kf_id]
            if src_kf_id > 0:
                supp_src_ids.append(src_kf_id - 1)
            
            # collect supporting stuff
            for s_src_kf_id in supp_src_ids:
                for s_trg_kf_id in range(len(supp_kfs[s_src_kf_id])):
                    next_kf = supp_kfs[s_src_kf_id][s_trg_kf_id]
                    next_kf_ts = supp_kf_ts[s_src_kf_id][s_trg_kf_id]
                    trg_images_precomputed[src_kf_id].append(next_kf.image)
                    trg_images_precomputed_ts[src_kf_id].append(next_kf_ts)

        if stack:
            for src_kf_id in range(len(kfs)):
                if len(trg_images_precomputed[src_kf_id]) > 0:
                    trg_images_precomputed[src_kf_id] = torch.stack(trg_images_precomputed[src_kf_id], dim=0).contiguous()
        torch.set_grad_enabled(True)

        return trg_images_precomputed, trg_images_precomputed_ts

    def mapping(self, num_iters=500, mode='map'):
        assert(mode in ['init', 'map', 'supp'])
        
        viz_message = 'mapping'
        if mode == 'supp':
            viz_message = 'supp_mapping'

        torch.cuda.synchronize()
        start_time = time.time()
        print('TRACKED KFS', self.tracked_timestamps)
        if mode == 'init':
            self.reset_running_supp_kfs()
            self.reset_tracked_poses()
        else:
            self.tracked_poses_to_supp()


        self.check_kf_integrity()
        print('STARTED MAPPING !!!!!!!!!!!!! on ', self.kf_timestamps, 
              'and supporting frames',             self.curr_supp_kf_timestamps)

        device = torch.device('cuda:0')

        mapping_params = self.setup_mapping_opt(mode, device) 
        optim = mapping_params['optimiser']
        kf_logdepths = mapping_params['logdepths']
        delta_poses = mapping_params['delta_poses']
        affine_comps = mapping_params['affine']
        
        # supplementarry frames 
        supp_delta_poses = mapping_params['supp_delta_poses']
        supp_affine_comps = mapping_params['supp_affine']

        kf_poses = [pose.detach().clone() for pose in self.kf_poses]

        supporting_info = self.get_supp_kf_poses_pairs()

        supp_kfs = supporting_info['kfs']
        supp_kf_poses = supporting_info['poses']
        supp_kf_ts = supporting_info['timestamps']

        kfs = self.kfs

        residual_mode = 'colour'

        cost_params = copy.deepcopy(self.config['aligment']['cost_params'])
        cost_params['mode'] = residual_mode
        cost_params['collect_stats'] = 2

        median_loss_weight = self.config['aligment']['median_loss_weight']
        print('map: num kfs', len(self.kfs), 'num curr supporting frames', len(self.curr_supp_kf_poses))
        prev_loss = torch.inf
        stop_tol = 1e-8
        abs_stop_tol = 1e-8
        converged = False

        kf_connectivity = self.generate_connectivity_batch(mode)
        trg_images_precomputed, trg_images_precomputed_ts = self.collect_target_frames(mode,
                                                                                       supp_kfs,
                                                                                       supp_kf_ts)
        
        viz_trg_images_precomputed, _ = self.collect_target_frames(mode,
                                                                    supp_kfs,
                                                                    supp_kf_ts,
                                                                    stack=False)
        
        for indx in range(len(viz_trg_images_precomputed)):
            viz_trg_images_precomputed[indx] = list_cpu(viz_trg_images_precomputed[indx])
        
        for iter in range(num_iters):
            self.check_if_paused()
            if converged:
                break
            residuals = []

            for src_kf_id, trg_kf_ids in kf_connectivity.items():

                curr_kf = kfs[src_kf_id]
                curr_kf_logdepth = kf_logdepths[src_kf_id]
                curr_kf_pose = kf_poses[src_kf_id]
                curr_kf_delta = delta_poses[src_kf_id]
                curr_kf_affine = affine_comps[src_kf_id]

                src_pose = curr_kf_pose
                src_delta = self.pose_to_mat(curr_kf_delta)
                
                # this one goes over other KFs 
                # collecting
                trg_Ks = []
                trg_d_poses = []
                trg_affines = []
                trg_ts = []
                for trg_kf_id in trg_kf_ids:
                    next_kf = kfs[trg_kf_id]
                    next_kf_pose = kf_poses[trg_kf_id]
                    next_kf_delta = delta_poses[trg_kf_id]
                    next_kf_affine = affine_comps[trg_kf_id]

                    trg_pose = next_kf_pose
                    if self.opt_supporting:
                        trg_delta = self.pose_to_mat(next_kf_delta)
                    else:
                        trg_delta = torch.eye(4, device=device)

                    # trg_images.append(next_kf.image)
                    trg_Ks.append(next_kf.K)
                    trg_d_poses.append(trg_delta @ torch.linalg.inv(trg_pose) @ src_pose @ torch.linalg.inv(src_delta))
                    # trg_d_poses.append(torch.linalg.inv(trg_pose) @ src_pose @ torch.linalg.inv(src_delta))
                    trg_affines.append(next_kf_affine)
                    trg_ts.append(self.kf_timestamps[trg_kf_id])

                supp_src_ids = [src_kf_id]
                if src_kf_id > 0:
                    supp_src_ids.append(src_kf_id - 1)

                for s_src_kf_id in supp_src_ids:
                    for trg_kf_id in range(len(supp_kfs[s_src_kf_id])):
                        # next_kf = supp_kfs[s_src_kf_id][trg_kf_id]
                        next_kf_pose = supp_kf_poses[s_src_kf_id][trg_kf_id]
                        next_kf_delta = supp_delta_poses[s_src_kf_id][trg_kf_id]
                        next_kf_affine = supp_affine_comps[s_src_kf_id][trg_kf_id]

                        trg_pose = next_kf_pose
                        if self.opt_supporting:
                            trg_delta = self.pose_to_mat(next_kf_delta)
                        else:
                            trg_delta = torch.eye(4, device=device)

                        # trg_images.append(next_kf.image)
                        trg_Ks.append(next_kf.K)
                        trg_d_poses.append(trg_delta @ torch.linalg.inv(trg_pose) @ src_pose @ torch.linalg.inv(src_delta))
                        # trg_d_poses.append(torch.linalg.inv(trg_pose) @ src_pose @ torch.linalg.inv(src_delta))
                        trg_affines.append(next_kf_affine)
                        trg_ts.append(supp_kf_ts[s_src_kf_id][trg_kf_id])
                

                # trg_images = torch.stack(trg_images, dim=0)
                trg_images = trg_images_precomputed[src_kf_id]

                trg_Ks = torch.stack(trg_Ks, dim=0)
                trg_d_poses = torch.stack(trg_d_poses, dim=0)
                if self.affine_compensation:
                    trg_affines = torch.stack(trg_affines, dim=0)
                # sanity check
                assert(trg_images_precomputed_ts[src_kf_id] == trg_ts)

                residuals_curr_to_next = dense_optim_batch.photomeric_cost_batch(curr_kf, 
                                                            trg_images, 
                                                            trg_Ks,
                                                            curr_kf_logdepth, 
                                                            poses=trg_d_poses,
                                                            affine_comp=(curr_kf_affine, trg_affines) if self.affine_compensation else None,
                                                            cost_config=cost_params)
                

                residuals.append(residuals_curr_to_next)
            

            losses_per_pair = []
            for src_id, residual in enumerate(residuals):
                assert(torch.all(torch.isnan(residual['residual']) == False))
                losses_per_pair.append((residual['residual']).mean())
                
            loss = torch.sum(torch.stack(losses_per_pair, dim=0))
            # depth_seeds = torch.exp(torch.cat(kf_logdepths, dim=0))
            # loss += median_loss_weight * depth_median_loss(kf_logdepths)

            assert(torch.all(torch.isnan(loss) == False))
            loss.backward()
            
            optim.step()
            optim.zero_grad()

            # handle poses
            with torch.no_grad():
                kf_deltas_saved = []
                for ind, delta in enumerate(delta_poses):
                    delta = self.pose_to_mat(delta)
                    kf_deltas_saved.append(delta.detach().clone())
                    kf_poses[ind] = kf_poses[ind] @ torch.linalg.inv(delta)
                    kf_poses[ind] = lie_algebra.renormalise_se3(kf_poses[ind])
                    # zero out delta 
                    delta_poses[ind] = zero_out_lietorch_tensor(delta_poses[ind])

                if not self.opt_supporting:
                    for src_id in range(len(self.kfs)):
                        for ind in range(len(supp_kf_poses[src_id])):
                            supp_kf_poses[src_id][ind] = supp_kf_poses[src_id][ind] @ torch.linalg.inv(kf_deltas_saved[src_id])
                else:
                    for src_id in range(len(self.kfs)):
                        for ind, delta in enumerate(supp_delta_poses[src_id]):
                            inv_delta = torch.linalg.inv(self.pose_to_mat(delta))
                            supp_kf_poses[src_id][ind] = supp_kf_poses[src_id][ind] @ inv_delta 
                            supp_kf_poses[src_id][ind] = lie_algebra.renormalise_se3(supp_kf_poses[src_id][ind])
                            # zero out delta 
                            supp_delta_poses[src_id][ind] = zero_out_lietorch_tensor(supp_delta_poses[src_id][ind])
                    


            # self.check_if_paused()

            with torch.no_grad():
                if iter % 100 == 0 and mode != 'supp': 
                    vis_supp_ts = []
                    vis_supp_poses = []
                    for vis_src_id in range(len(self.kfs)):
                        for vis_trg_id in range(len(supp_kf_poses[vis_src_id])):
                            vis_supp_poses.append(supp_kf_poses[vis_src_id][vis_trg_id])
                            vis_supp_ts.append(supp_kf_ts[vis_src_id][vis_trg_id])

                    self.viz_queue.push((viz_message,
                                        ([attrs_on_cpu(kf) for kf in kfs], viz_trg_images_precomputed),
                                        self.kf_timestamps + vis_supp_ts,
                                        list_cpu(kf_poses) + list_cpu(vis_supp_poses),
                                        [dict_cpu(residuals_opt) for residuals_opt in residuals],
                                        kf_connectivity,
                                        ), block=False)



            if self.initialised:
                abs_loss = abs(loss.item() - prev_loss)
                rel_loss = abs_loss / prev_loss
                if rel_loss < stop_tol:
                    converged = True
                    print('early stopping')
                    break

                prev_loss = loss.item()

        if not (mode == 'supp' and self.mapping_scheduled):
            vis_supp_ts = []
            vis_supp_poses = []
            for vis_src_id in range(len(self.kfs)):
                for vis_trg_id in range(len(supp_kf_poses[vis_src_id])):
                    vis_supp_poses.append(supp_kf_poses[vis_src_id][vis_trg_id])
                    vis_supp_ts.append(supp_kf_ts[vis_src_id][vis_trg_id])
            
            self.viz_queue.push((viz_message,
                                ([attrs_on_cpu(kf) for kf in kfs], viz_trg_images_precomputed),
                                self.kf_timestamps + vis_supp_ts,
                                list_cpu(kf_poses) + list_cpu(vis_supp_poses),
                                [dict_cpu(residuals_opt) for residuals_opt in residuals],
                                kf_connectivity,
                                ), block=False)
        
        # todo assign new poses and logdepths
        print('Mapping done, updating keyframes')
        torch.cuda.synchronize() 
        time_taken = time.time() - start_time
        print(f'Mapping time {time_taken:.3f} sec')
            
        # update with newly mapped parameters
        self.kf_logdepths = [logdepth.detach().clone() for logdepth in kf_logdepths]
        delta_last_kf = kf_poses[-1] @ torch.linalg.inv(self.kf_poses[-1])
        delta_last_kf = delta_last_kf.detach().clone()

        self.kf_poses = [pose.detach().clone() for pose in kf_poses]
        self.kf_affines = [affine.detach().clone() for affine in affine_comps]

        for src_id in range(len(self.kfs)):
            if not self.initialised:
                continue

            for indx in range(len(self.supp_kfs_opt[src_id])):
                pose = supp_kf_poses[src_id][indx].detach().clone()
                affine = supp_affine_comps[src_id][indx].detach().clone()
                if src_id == len(self.kfs) - 1:
                    self.curr_supp_kf_poses[indx] = pose
                    self.curr_supp_kf_affines[indx] = affine
                else:
                    self.supp_kfs_opt[src_id][indx].update_params(pose, affine)
        
        for new_kf_pose, kf_timestamp in zip(self.kf_poses, self.kf_timestamps):
            self.global_kf_trajectory[kf_timestamp] = to_np(new_kf_pose)
            self.global_kf_scale[kf_timestamp] = 1.0

        self.update_track_pose(mode)
    
        self.initialised = True
        return

    def update_track_pose(self, mode):
        latest_map_ts = int(self.kf_timestamps[-1])

        if len(self.curr_supp_kf_timestamps) == 0 or latest_map_ts > (int(self.curr_supp_kf_timestamps[-1])):
            assert(mode != 'supp')
            self.current_track = self.kf_poses[-1].detach().clone()
            if self.affine_compensation:
                self.current_aff = self.kf_affines[-1].detach().clone()
            self.current_ts = self.kf_timestamps[-1]
        else:
            self.current_track = self.curr_supp_kf_poses[-1].detach().clone()
            if self.affine_compensation:
                self.current_aff = self.curr_supp_kf_affines[-1].detach().clone()
            self.current_ts = self.curr_supp_kf_timestamps[-1]
        
        return

    def is_kf(self, frame, pose, timestamp):
        num_init_frames = self.config['aligment']['init_frames']
        if int(timestamp) - int(self.start_id) <  num_init_frames and not self.initialised:
            return False, None
        if int(timestamp) - int(self.start_id) == num_init_frames and not self.initialised:
            return True, None
        
        self.depth_validity_ratio = self.config['kf']['depth_validity_ratio']
        self.translation_diff = self.config['kf']['translation_thresh']
        
        self.rotation_diff = -1
        if 'rotation_thresh' in self.config['kf']:
            self.rotation_diff = self.config['kf']['rotation_thresh']
        is_kf = False        

        pose = pose.detach().clone()
        est_depth = self.estimate_depth_latest_kf(pose)
        valid_depth = est_depth > 1e-6
        validity_ratio = valid_depth.sum() / valid_depth.nelement()

        pose_diff, scale = translation_difference(pose, self.kf_poses[-1], est_depth)

        if validity_ratio < self.depth_validity_ratio:
            is_kf = True
            print('new kf: low depth validity ratio {}'.format(validity_ratio))
    
        if pose_diff > self.translation_diff:
            is_kf = True
            print('new kf: large pose diff {}'.format(pose_diff))

        return is_kf, scale
        
    def run(self):
        load = 'restore' in self.config
        if load:
            self.load_state(self.config['restore']['path'], 
                            self.config['restore']['frame_id'])
        else:
            self.raw_init()

        self.mapping_params = self.config['aligment']['mapping']

        self.kf_queue.push(('frame', self.start_id))

        for frame_id in range(self.start_id + 1, len(self.dataset)):
            self.kf_queue.push(('frame', frame_id))
            self.current_frame_id = frame_id

            frame = self.dataset[frame_id]
            timestamp = self.id_to_timestamp(frame_id)
            track_result = self.track_frame(frame, timestamp)

            if self.initialised:
                num_supp_mapping = self.mapping_params['continual_steps']
                if num_supp_mapping > 0:
                    print('Running supplementary mapping')
                    self.mapping(num_supp_mapping, mode='supp')

            self.check_if_paused()

            if self.mapping_scheduled:
                # check that we have enough supporting frames
                if len(self.curr_supp_kfs) >= 2:
                    print('Running scheduled mapping')
                    num_map_iters = self.mapping_params['steps']
                    self.mapping(num_map_iters, mode='map')
                    self.mapping_scheduled = False
                    self.reset_tracked_poses()
                    self.reset_running_supp_kfs()
                    print('mapping done')

            # assert we didn't mess up tracking timestamps
            assert(track_result['ts'] == self.current_ts)
            is_kf, kf_depth_scale = self.is_kf(frame, self.current_track, timestamp)
            
            if is_kf:
                self.flush_tracked_poses_to_supp()
                self.init_keyframe(frame, self.current_track, timestamp, self.current_aff)
                self.reset_tracked_poses()
                self.reset_running_supp_kfs()

                if not self.initialised:
                    print('initialising mapping')
                    self.reset_tracked_poses()
                    self.reset_running_supp_kfs()
                    self.mapping(self.mapping_params['init_steps'], mode='init')
                else:   
                    self.mapping_scheduled = True
                    print('new keyframe initialised, scheduled mapping')

            if frame_id > 0 and self.save_every > 0 and frame_id % self.save_every == 0:
                self.save_state()

            # frame_id = self.frame_queue.get(block=True)
        print('optimisation done')

        self.save_traj('final')
        if self.save_every > 0:
            # self.save_traj('final')
            self.save_state()
        
        self.viz_queue.push(("end",))
        self.waitev.wait()

    def load_state(self, state_path, frame_id):
        self.init_system()
        self.reset_kfs()
        self.reset_running_supp_kfs()
        self.reset_tracked_poses()
        
        if isinstance(frame_id, int):
            frame_id = self.id_to_timestamp(frame_id)

        # a function to fully restore the state of the system
        path = Path(state_path)
        assert(path.exists())
        kf_dir = path / 'curr_kfs' / frame_id
        traj_dir = path / 'traj'

        assert(kf_dir.exists())
        # gather all kf files matching pattern kf_*.pkl,
        kfs_files = list(kf_dir.glob('kf_*.pkl'))
        kfs_timestamps = [kf_file.stem.split('_')[1] for kf_file in kfs_files]
        kfs_timestamps = sorted(kfs_timestamps, key=lambda x: int(x))
        
        if len(kfs_timestamps) > self.window_size:
            # loading only last window_size kfs
            kfs_timestamps = kfs_timestamps[-self.window_size:]

        for ts in kfs_timestamps:
            kf, kf_pose, kf_logdepth, kf_affine, kf_timestamp = load_kf(kf_dir, ts)
            self.add_kf(kf=kf,
                        pose=kf_pose,
                        keypoints_logdepth=kf_logdepth,
                        timestamp=kf_timestamp,
                        aff=kf_affine)
        print('loaded kfs', self.kf_timestamps)
        self.update_track_pose(mode='init')
        self.initialised = True

        self.current_frame_id = int(self.kf_timestamps[-1])
        self.start_id = self.current_frame_id + 1

        kf_traj_path = traj_dir / f'kf_traj_{frame_id}.pkl'
        track_traj_path = traj_dir / f'track_traj_{frame_id}.pkl'
        assert(kf_traj_path.exists())
        assert(track_traj_path.exists())
        
        with open(kf_traj_path, 'rb') as f:
            self.global_kf_trajectory = pickle.load(f)
            print('loaded kf positions from', kf_traj_path)
        with open(track_traj_path, 'rb') as f:
            self.global_track_trajectory = pickle.load(f)
            print('loaded track positions from', track_traj_path)

        kf_scales_path = traj_dir / f'kf_traj_scales_{frame_id}.pkl'
        if kf_scales_path.exists():
            with open(kf_scales_path, 'rb') as f:
                self.global_kf_scale = pickle.load(f)
                print('loaded kf scales from', kf_scales_path)
        else:
            print('kf scales not found, using 1.0')
            self.global_kf_scale = {ts: 1.0 for ts in self.global_kf_trajectory.keys()}

        cpu_kfs = [attrs_on_cpu(self.kfs[-1])]
        timestamps = [self.kf_timestamps[-1]]
        cpu_residuals = [dict_cpu(dense_optim.unproject_kf(self.kfs[-1], self.kf_logdepths[-1]))] 
        
        traj_info = ( dict_cpu(self.global_kf_trajectory), dict_cpu(self.global_track_trajectory),)
        self.viz_queue.push(('init',
                            cpu_kfs,
                            timestamps,
                            list_cpu([self.kf_poses[-1]]),
                            cpu_residuals,
                            traj_info,
                            ), block=True)
        return

    def save_state(self, name=None):
        print(f'Saving systems state at {self.current_frame_id}')
        save_path = self.save_path
        # save_path = Path(self.config['save_path'])
        current_step = self.current_frame_id
        # format with 6 digits
        current_step = str(current_step).zfill(6)
        if name is not None:
            current_step = name

        kf_save_path = save_path / 'curr_kfs' / current_step 
        kf_save_path.mkdir(parents=True, exist_ok=True)

        for kf, kf_pose, kf_logdepth, kf_affine, kf_timestamp in zip(self.kfs, 
                                                                     self.kf_poses,
                                                                     self.kf_logdepths,
                                                                    #  self.kf_scales,
                                                                     self.kf_affines,
                                                                     self.kf_timestamps):
            dump_kf(kf_save_path, kf, kf_pose, kf_logdepth, kf_affine, kf_timestamp)
        
        self.save_traj(current_step)
        
        # dump global trajectory
        return 
    
    def save_traj(self, current_step):
        save_path = self.save_path
        traj_save_path = save_path / 'traj'
        traj_save_path.mkdir(parents=True, exist_ok=True)
        kf_traj_path = traj_save_path / f'kf_traj_{current_step}.pkl'
        track_traj_path = traj_save_path / f'track_traj_{current_step}.pkl'
        kf_scales_path = traj_save_path / f'kf_traj_scales_{current_step}.pkl'
        with open(kf_traj_path, 'wb') as f:
            pickle.dump(self.global_kf_trajectory, f)
        with open(track_traj_path, 'wb') as f:
            pickle.dump(self.global_track_trajectory, f)
        with open(kf_scales_path, 'wb') as f:
            pickle.dump(self.global_kf_scale, f)
        print('Saved systems state')
        return
    
    def init_dirs(self):
        self.save_path = Path(self.config['save_path'])
        # modify it to save_path_datetime
        datetime = time.strftime("%Y_%m_%d@%H_%M_%S", time.localtime())
        self.save_path = self.save_path.parent / (self.save_path.name + '_' + datetime)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        # copy config file to save path
        config_save_path = self.save_path / 'config.yaml'
        # copy from self.config_path
        shutil.copyfile(self.config_path, config_save_path)            
    
        print('CURRENT SAVE PATH:', self.save_path)
        self.kf_save_path = self.save_path / 'kfs'
        if not self.kf_save_path.exists():
            self.kf_save_path.mkdir(parents=True)   

    def add_kf(self, kf, pose, keypoints_logdepth, timestamp, aff=None, indx=None):
        if indx is None:
            self.kfs.append(kf)
            self.kf_poses.append(pose)
            self.kf_logdepths.append(keypoints_logdepth)
            # self.kf_scales.append(kf_scale)
            self.kf_timestamps.append(timestamp)
            self.kf_affines.append(aff)

            self.supp_kfs_class.append([])
            self.supp_kfs_opt.append([])            
        else:
            self.kfs.insert(indx, kf)
            self.kf_poses.insert(indx, pose)
            self.kf_logdepths.insert(indx, keypoints_logdepth)
            # self.kf_scales.insert(indx, kf_scale)
            self.kf_timestamps.insert(indx, timestamp)
            self.kf_affines.insert(indx, aff)

            self.supp_kfs_class.insert(indx, [])
            self.supp_kfs_opt.insert(indx, [])
        return
        
    def pop_kf(self, indx):
        self.kfs.pop(indx)
        self.kf_poses.pop(indx)
        self.kf_logdepths.pop(indx)
        # self.kf_scales.pop(indx)
        self.kf_timestamps.pop(indx)
        self.kf_affines.pop(indx)

        self.supp_kfs_class.pop(indx)
        self.supp_kfs_opt.pop(indx)

    def check_kf_integrity(self):
        assert(len(self.kfs) == len(self.kf_poses))
        assert(len(self.kfs) == len(self.kf_logdepths))
        assert(len(self.kfs) == len(self.kf_timestamps))
        # assert(len(self.kfs) == len(self.kf_scales))
        assert(len(self.kfs) == len(self.kf_affines) or not self.affine_compensation)

        assert(len(self.curr_supp_kfs) == len(self.curr_supp_kf_poses))
        assert(len(self.curr_supp_kfs) == len(self.curr_supp_kf_timestamps))
        assert(len(self.curr_supp_kfs) == len(self.curr_supp_kf_affines) or not self.affine_compensation)

        assert(len(self.supp_kfs_class) == len(self.supp_kfs_opt))
        assert(len(self.supp_kfs_class) == len(self.kfs))

    def tracked_poses_to_supp(self,):
        if not self.initialised:
            # no supporting for initialisation ?
            self.reset_tracked_poses()
            self.reset_running_supp_kfs()
            return
    
        self.reset_running_supp_kfs()

        supp_kfs, supp_kfs_opts = self.collect_tracking_frames(last=True)
        
        for kf, kf_opt in zip(supp_kfs, supp_kfs_opts):
            assert(kf.timestamp == kf_opt.timestamp)
            self.curr_supp_kfs.append(kf.kf)
            self.curr_supp_kf_poses.append(kf_opt.pose)
            self.curr_supp_kf_timestamps.append(kf_opt.timestamp)
            self.curr_supp_kf_affines.append(kf_opt.affine)

        return 
    
    def running_supp_to_newformat(self):
        if not self.initialised:
            return [], []
        
        if self.affine_compensation:
            supp_kf_affines = self.curr_supp_kf_affines
        else:
            supp_kf_affines = [None for i in range(len(self.curr_supp_kfs))]

        
        supp_kfs = []
        supp_kfs_opts = []

        for frame, pose, ts, aff in zip(self.curr_supp_kfs, 
                                        self.curr_supp_kf_poses, 
                                        self.curr_supp_kf_timestamps, 
                                        supp_kf_affines):
            print('Converting supp kf', ts)
            supp_kfs.append(SupportingKF(kf=frame, timestamp=ts))
            supp_kfs_opts.append(ParamsSupportingKF(pose=pose, timestamp=ts, affine=aff))

        return supp_kfs, supp_kfs_opts

    def flush_tracked_poses_to_supp(self,):
        # THIS IS suppoused to be called after adding new kf
        # select at most two
        supp_kfs, supp_kfs_opts = self.collect_tracking_frames(last=False)

        assert(len(self.supp_kfs_opt[-1]) == 0)
        assert(len(self.supp_kfs_class[-1]) == 0)

        self.supp_kfs_class[-1] = supp_kfs
        self.supp_kfs_opt[-1] = supp_kfs_opts

        return 
    
    def collect_tracking_frames(self, last=False):
        # select frames from the current tracked pool
        n_tr_frames = len(self.tracked_frames)
        if last:
            frame_ids = [n_tr_frames - 1, n_tr_frames - 2]
        else:
            each_n = self.config['aligment']['mapping']['supp_every_n']
            frame_ids = [i * (n_tr_frames - 1) // each_n + 1 for i in range(1, each_n)]

        # filter duplicates
        to_add_ids = sorted(set(frame_ids))
        ids = []
        # filter negative values
        for id in to_add_ids:
            if id >= 0 and id < n_tr_frames:
                ids.append(id)
        
        collected_frames = [self.tracked_frames[i] for i in ids]
        collected_poses = [self.tracked_poses[i] for i in ids]
        collected_timestamps = [self.tracked_timestamps[i] for i in ids]
        if self.affine_compensation:
            collected_affines = [self.tracked_affines[i] for i in ids]
        else:
            collected_affines = [None for i in ids]
        
        supp_kfs = []
        supp_kfs_opts = []
        for frame, pose, ts, aff in zip(collected_frames, 
                                        collected_poses, 
                                        collected_timestamps, 
                                        collected_affines):
            supp_kfs.append(SupportingKF(kf=frame, timestamp=ts))
            supp_kfs_opts.append(ParamsSupportingKF(pose=pose, timestamp=ts, affine=aff))
        return supp_kfs, supp_kfs_opts
    
    def reset_tracked_poses(self):
        # before resetting we want to save some tracking frames to new supporting frames
        self.tracked_frames = []
        self.tracked_poses = []
        self.tracked_timestamps = []
        self.tracked_affines = []

    def reset_running_supp_kfs(self):
        # these are current supporting frames using for the latest keyframe in mapping
        self.curr_supp_kfs = []
        self.curr_supp_kf_poses = []
        self.curr_supp_kf_timestamps = []
        self.curr_supp_kf_affines = []

    def reset_new_supp_kfs(self):
        self.supp_kfs_class = [[] for i in range(len(self.kfs))]
        self.supp_kfs_opt = [[] for i in range(len(self.kfs))]
    
    def pop_new_supp_kfs(self, indx=0):
        self.supp_kfs_class.pop(indx)
        self.supp_kfs_opt.pop(indx)

    def reset_kfs(self):
        self.kfs = []
        self.kf_poses = []
        self.kf_logdepths = []
        # self.kf_scales = []
        self.kf_timestamps = []
        self.kf_affines = []

    def check_if_paused(self):
        try:
            is_paused = self.pause_queue.get(block=False)
            self.paused = is_paused
        except queue.Empty:
            pass

        if self.paused:
            while self.paused:
                try:
                    is_paused = self.pause_queue.get(block=False)
                    self.paused = is_paused
                except queue.Empty:
                    pass
                time.sleep(0.0001) 
        return
            