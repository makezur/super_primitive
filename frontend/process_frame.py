from torch import nn
import torch
import torch.nn.functional as F 

import frontend.segment.sam_tools as sam_tools
import copy

import frontend.segment.mask_generation as mask_generation
from frontend.normals.normals_inferer import setup_normals_predictor, predict_normals
import tool.camera as camera

import frontend.normals.normals_integration as normals_integration
from frontend.segment.post_processer import kf_fix_disconnected_regions

from image.keyframe import KeyFrame, put_keypoints_back

from tool.etc import image_tt, to_img_np

def setup_new_front_processor(config, device='cuda:0'):
    model_normals = setup_normals_predictor(config['frontend']['normals_path'], scannet=config['frontend']['normals_scannet'])
    model_sam = sam_tools.setup_sam(config['frontend']['sam_path'], device=device)

    return FrontProcessorNew(model_normals, model_sam, config=config)


def resize_masks(masks, spatial_dim):
    return F.interpolate(masks.float()[:, None],
                         size=spatial_dim,
                         mode='nearest')[:, 0].bool()

class FrontProcessorNew(nn.Module):
    def __init__(self, model_normals, model_sam,
                 config) -> None:
        super().__init__()

        self.model_normals = model_normals
        self.model_sam = model_sam

        self.config = copy.deepcopy(config)

        # dimension in which the normals network expects the image
        self.normal_resize = self.config['frontend']['normals']['network_dim']
        self.normal_int_shape = self.config['frontend']['normals']['integration_shape']

        self.infer_sam_resolution = None
        if 'infer_resolution' in self.config['sam_params']:
            self.infer_sam_resolution = self.config['sam_params']['infer_resolution']

        self.integration_mode = self.config['frontend']['normals']['integrator']
        assert(self.integration_mode in ['bini', 'tiled'])

        self.num_pts = self.config['frontend']['num_pts']
        self.num_pts_active = self.config['frontend']['num_pts_active']
        self.check_for_depth_disc = False
        if 'check_for_depth_disc' in self.config['frontend']:
            self.check_for_depth_disc = self.config['frontend']['check_for_depth_disc']
            self.depth_disc_params = self.config['frontend']['depth_disc_params']

        self.include_normals = self.config['frontend']['include_normals']

        self.pool_edges = False

        self.verbose = False

    def predict_normals(self, image):
        torch.set_grad_enabled(False)
        norm_out = predict_normals(self.model_normals, image,
                                   resize=True, resize_ratio=self.normal_resize,
                                   resize_back=False)

        pred_norm = norm_out[:, :3, :, :]
        pred_norm = pred_norm.detach().clone().permute(0, 2, 3, 1)
        pred_kappa = norm_out[:, 3:, :, :]

        torch.set_grad_enabled(True)
        return pred_norm, pred_kappa
    
    def integrate_normals_tile(self, masks, normals, K):
        if self.verbose:
            print('FRONTEND: integrating normals in tiled-style', flush=True)

        assert(normals.shape[0] == 1)
        integrated_depth = normals_integration.run_tiled_normal_integration(normals[0],
                                                                            K, 
                                                                            masks,
                                                                            down_scale=1,
                                                                            cg_max_iter=self.config['frontend']['cg_max_iter'],
                                                                            cg_tol=self.config['frontend']['cg_tol'])
        if self.verbose:
            print('FRONTEND: integrated depth is of shape', integrated_depth.shape, flush=True)

        return integrated_depth, normals
    
    def infer_masks(self, image, int_normal_shape, keypoints=None):
        H, W, _ = image.shape
        image_spatial = (H, W)
        if self.infer_sam_resolution is not None:
            if self.verbose:
                print('SAM inference: interpolating image to', 
                    self.infer_sam_resolution, flush=True)
            image = image_tt(image, 'cpu')
            image = F.interpolate(image[None], 
                                  size=self.infer_sam_resolution, 
                                  mode='bilinear')[0]
            image = to_img_np(image)
        
        generated_masks = mask_generation.infer_masks(self.model_sam,
                                                      image,
                                                      sam_config=self.config['sam_params'],
                                                      keypoints=keypoints,
                                                      num_pts=self.num_pts,
                                                      num_pts_active=self.num_pts_active,
                                                      edge_probs_shape=int_normal_shape)
        if self.infer_sam_resolution is not None:
            # downsample stuff back
            generated_masks['masks']['masks'] = resize_masks(generated_masks['masks']['masks'], 
                                                             image_spatial)
        return generated_masks

    
    def preprocess(self, image, K, keypoints=None, normals=None):
        torch.set_grad_enabled(False)
        if normals is None:
            pred_norm, pred_kappa = self.predict_normals(image)
            pred_norm_og_size = pred_norm.detach()
        else:
            print('Warning! using GT normals')
            pred_norm = normals

        H, W, _ = image.shape
        H_geom, W_geom = self.normal_int_shape
        K_geom = camera.instrinsic_scaled_K_anisotropic(K, 
                                                    scale_H=H_geom / H, 
                                                    scale_W=W_geom / W)
        int_normal_shape = (H_geom, W_geom)

        if pred_norm.shape[1] != H_geom or pred_norm.shape[2] != W_geom:
            pred_norm = pred_norm.permute(0, 3, 1, 2)
            pred_norm = F.interpolate(pred_norm, 
                                    size=int_normal_shape, 
                                    mode='nearest')
            pred_norm = pred_norm.permute(0, 2, 3, 1)

        # run segmentation front end
        generated_masks = self.infer_masks(image, int_normal_shape)
        
        torch.cuda.synchronize()
        masks_coarse = torch.nn.functional.interpolate(generated_masks['masks']['masks'].float()[:, None],
                                                        size=(H_geom, W_geom),
                                                        mode='nearest')[:, 0]

        integrated_depth, normals = self.integrate_normals_tile(masks_coarse, 
                                                                pred_norm, 
                                                                K_geom)

        device = generated_masks['masks']['masks'].device
        results = {'pred_norm': pred_norm_og_size,
                   'pred_norm_kappa': pred_kappa,
                   'masks_coarse': masks_coarse,
                   'normals_processed': normals,
                   'integrated_depth': integrated_depth}
        
        # results.update(supp_info)
        results.update(generated_masks)
        if self.verbose:
            print('FRONTEND: Finished processing frame', flush=True)
        torch.set_grad_enabled(True)
        return results
    
    def _downsample_to_target(self, image_torch, K):
        target_kf_size = self.config['frontend']['downsample_pow']
        
        H, W = image_torch.shape[1:]
        H_kf = H // (2 ** target_kf_size)
        W_kf = W // (2 ** target_kf_size)

        K_kf = camera.instrinsic_scaled_K_anisotropic(K, 
                                                            scale_H=H_kf / H, 
                                                            scale_W=W_kf / W)
        
        image_torch = F.interpolate(image_torch[None], 
                                    size=(H_kf, W_kf), 
                                    mode='bilinear')[0]
        
        result = {'image_torch': image_torch,
                 'K_kf': K_kf,
                 'H_kf': H_kf,
                 'W_kf': W_kf}
        return result
    
    def to_final_image(self, image_torch, normals=None, normals_kappa=None):
        if not self.include_normals:
            return image_torch
        H, W = image_torch.shape[1:]
        normals_downsampled = torch.nn.functional.interpolate(normals.permute(0, 3, 1, 2),
                                                              size=(H, W),
                                                              mode='nearest')[0]

        kappa_downsampled = torch.nn.functional.interpolate(normals_kappa,
                                                            size=(H, W),
                                                            mode='nearest')[0]

        image_torch =  torch.cat([image_torch, normals_downsampled], dim=0)

        return image_torch 
    
    def process_to_kf(self, image, K, keypoints=None, return_raw=False):
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()

        preprocessed = self.preprocess(image, K, keypoints)

        device = preprocessed['keypoints'].device


        image_torch = image_tt(image, device)
        K = torch.from_numpy(K).to(device).float()

        downsampled = self._downsample_to_target(image_torch, K)

        if self.include_normals:
            downsampled['image_torch'] = self.to_final_image(downsampled['image_torch'],
                                                             preprocessed['pred_norm'],
                                                             preprocessed['pred_norm_kappa'])
        image_torch = downsampled['image_torch']

        K_kf = downsampled['K_kf']
        H_kf, W_kf = downsampled['H_kf'], downsampled['W_kf']


        logdepth = torch.nn.functional.interpolate(preprocessed['integrated_depth'][:, None],
                                                   size=(H_kf, W_kf),
                                                   mode='nearest')[:, 0]   
        masks = logdepth > 1e-7
        keypoints, masks, logdepth = put_keypoints_back(preprocessed['keypoints'], masks, logdepth)
        logdepth[masks] = torch.log(logdepth[masks])

        torch.cuda.empty_cache()

        kf = KeyFrame(image_torch,
                      K=K_kf,
                      logdepth_perseg=logdepth,
                      keypoints=keypoints,
                      keypoint_regions=masks)
    
        if self.check_for_depth_disc:
            kf = kf_fix_disconnected_regions(kf,
                                             self.depth_disc_params['filter_size'],
                                             self.depth_disc_params['depth_threshold'],
                                             self.depth_disc_params['area_keep_ratio'])
        
        torch.set_grad_enabled(True)
        if return_raw:
            return kf, preprocessed
        return kf
    
    def process_to_supp_kf(self, image, K, device='cuda:0'):
        image_t = image_tt(image, device)
        K = torch.from_numpy(K).to(device).float()

        downsampled = self._downsample_to_target(image_t, K)

        if self.include_normals:
            pred_norm, pred_norm_kappa = self.predict_normals(image)
            downsampled['image_torch'] = self.to_final_image(downsampled['image_torch'],
                                                             pred_norm,
                                                             pred_norm_kappa)

        return KeyFrame(downsampled['image_torch'],
                        K=downsampled['K_kf'].float())