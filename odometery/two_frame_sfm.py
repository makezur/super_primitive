import torch
import torch.multiprocessing as mp

import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import core.dense_optim as dense_optim

import lietorch
import lie.lie_algebra as lie_algebra

import numpy as np

import image.keyframe as keyframe 
import frontend.process_frame as process_frame

import tool.point_utils as point_utils

import queue
import time
import copy 
import data 

from tool.etc import attrs_on_cpu, dict_cpu, list_cpu
import numpy as np

import yaml


class SfM(mp.Process):
    def __init__(self, config_path, waitev):
        super().__init__()

        self.waitev = waitev
        self.segment_id = 0

        self.opt_pose = True 

        self.config_path = config_path
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        self.src_id = self.config['dataset']['source_id']
        self.trg_id = self.config['dataset']['target_id']

        self.supp_ids = []


    def init_keyframes(self):
        dataset = data.load_dataset(self.config)
        front_processor = process_frame.setup_new_front_processor(self.config)
        src = dataset[self.src_id]
        trg = dataset[self.trg_id]

        self.src = src
        self.trg = trg

        torch.set_grad_enabled(False)

        self.front_processor = front_processor
        self.src_keyframe = front_processor.process_to_kf(src['image'], src['intrinsics'])

        self.supp_frames = [self.init_supporting_frame(trg)]

        for supp_id in self.supp_ids:
            supp = dataset[supp_id]
            self.supp_frames.append(self.init_supporting_frame(supp))

        torch.set_grad_enabled(True)
        self.paused = self.config['paused']

    def init_supporting_frame(self, frame):
        pose_gt = np.linalg.inv(frame['T']) @ self.src['T']

        if self.opt_pose:
            tq = lie_algebra.pose_to_tq(pose_gt[None])
            current_T = lietorch.SE3.InitFromVec(torch.from_numpy(tq).to('cuda:0').float())

            # add noise to the gt pose
            noise_T = lietorch.SE3.Random(1, sigma=0.05, device='cuda:0')
            current_T = current_T.mul(noise_T) 
            # current_T = lietorch.SE3.Identity(1).to(torch.device('cuda:0'))
            current_T = lietorch.LieGroupParameter(current_T)
            pose_to_mat = lambda x: x.retr().matrix()[0]
        else:
            current_T = pose_gt
            current_T = torch.from_numpy(current_T).to('cuda:0').float()
            pose_to_mat = lambda x: x
        

        supp_f = self.front_processor.process_to_supp_kf(frame['image'], frame['intrinsics'], device='cuda:0')

        return supp_f, current_T, pose_to_mat
    
    def init_optimisation(self):
        src_gt_depth_keypoints = self.src['depth']
        src_gt_depth_keypoints = torch.from_numpy(src_gt_depth_keypoints.copy()).to('cuda:0')
        self.src_keypoints = point_utils.denormalise_coordinates(self.src_keyframe.keypoints, self.src['image'].shape[:2])

        src_gt_depth_keypoints = src_gt_depth_keypoints[self.src_keypoints[:, 0], self.src_keypoints[:, 1]]

        # random depth seeds 
        src_depth_keypoints_opt = 2.0 + 2 * torch.rand_like(src_gt_depth_keypoints)

        src_depth_keypoints_opt = torch.log(src_depth_keypoints_opt) 
        src_gt_depth_keypoints = torch.log(src_gt_depth_keypoints)


        src_depth_keypoints_opt = nn.Parameter(src_depth_keypoints_opt)

        self.src_gt_depth_keypoints = src_gt_depth_keypoints
        self.src_depth_keypoints_opt = src_depth_keypoints_opt

        self.instatiate_optimisation()

    def instatiate_optimisation(self):
        self.adam_params = [{'params': self.src_depth_keypoints_opt, 'lr': 1e-3},
                            {'params': [pose for _, pose, _ in self.supp_frames], 
                             'lr': 1e-2}]

        optim = torch.optim.Adam(self.adam_params, lr=1e-3)

        self.optim = optim
        return 


    def run(self):
        num_iters = 500

        self.init_keyframes()
        self.init_optimisation()


        self.kf_queue.push((self.src_keyframe,), block=True) 
        src_keyframe_pyr = keyframe.keyframe_pyramid(self.src_keyframe, self.config['aligment']['pyramid_min'], 
                                                                        self.config['aligment']['pyramid_max'])
        support_frames_pyrs = [keyframe.keyframe_pyramid(supp_f, self.config['aligment']['pyramid_min'], 
                                                                 self.config['aligment']['pyramid_max']) for supp_f, _, _ in self.supp_frames]
        

        torch.cuda.empty_cache()         

        count = 0
        residual_mode = 'colour'
        cost_params = copy.deepcopy(self.config['aligment']['cost_params'])
        cost_params['mode'] = residual_mode
        cost_params['collect_stats'] = 2


        for pyr_level in range(len(src_keyframe_pyr)):

            src_keyframe_curr = src_keyframe_pyr[pyr_level]
            support_frames_curr = [supp_f_pyr[pyr_level] for supp_f_pyr in support_frames_pyrs]

            for i in range(num_iters):
                residuals_multi = []
                poses = []
                losses = []
                
                # go over support frames
                for supp_frame_id in range(len(support_frames_curr)):
                    supp_f_curr = support_frames_curr[supp_frame_id]
                    current_T = self.supp_frames[supp_frame_id][1]
                    pose_to_mat = self.supp_frames[supp_frame_id][2]

                    residuals_opt = dense_optim.photomeric_cost(src_keyframe_curr, supp_f_curr, 
                                                       self.src_depth_keypoints_opt, 
                                                       pose=pose_to_mat(current_T),
                                                       cost_config=cost_params)
                    losses.append(residuals_opt['residual'])
                    residuals_multi.append(residuals_opt)
                    poses.append(pose_to_mat(current_T).detach().clone())


                cpu_residuals = [dict_cpu(residuals_opt) for residuals_opt in residuals_multi]
                cpu_supp_frames = [attrs_on_cpu(supp_f_curr) for supp_f_curr in support_frames_curr]

                self.viz_queue.push((
                                     cpu_residuals,
                                     list_cpu(poses),
                                     attrs_on_cpu(src_keyframe_curr), 
                                     cpu_supp_frames
                                     ), block=False)

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
                        time.sleep(0.001)


                losses = [torch.mean(torch.abs(residuals_opt['residual'])) for residuals_opt in residuals_multi]     
                loss = torch.sum(torch.stack(losses))
                if count > 0:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                count += 1

                del residuals_multi, losses

        print('optimisation done')
        self.viz_queue.push(("end",))

        self.waitev.wait()

