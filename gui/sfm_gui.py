import torch # Must import before Open3D when using CUDA!
import torch.multiprocessing as mp

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import cv2 

import tool.etc as etc

import time
import threading

# import viz
import tool.viz as viz 
from tool.viz import render_pcd 
from tool.camera import instrinsic_scaled_K


from tool.multiprocess import TupleTensorQueue, release_data

import torch.utils.dlpack
import tool.o3d_tools as o3d_tools

from odometery.two_frame_sfm import SfM
import tool.point_utils as point_utils
import data
import yaml

from tool.camera import get_translation_norm, renorm_translation, apply_scale

def o3t_from_t(t):
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(t))

def enable_widget(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

def frustum_lineset(intrinsics, img_size, pose, scale=0.2):
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        img_size[1], img_size[0], intrinsics, 
        np.linalg.inv(pose), scale=scale)
    return frustum

def transform_points(pts, T):
    pts = np.einsum('ij, nj -> ni', T[:3, :3], pts)  + T[:3,3]
    return pts


class SfMWindow():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        self.viz_device = torch.device("cuda:0")
        self.viz_dtype = torch.float32


        self.window = gui.Application.instance.create_window(
            'Monocular SfM', width=1920, height=1080) # 1280 x 720 for recording
        em = 10

        self.window_residual = gui.Application.instance.create_window(
            'Residual Reprojected', width=1024 // 3 * 4, height=768 // 3)
        
        self.trg_image_widget = gui.ImageWidget()
        self.residual_widget = gui.ImageWidget()
        self.residual_widget_clicker = gui.ImageWidget() 
        self.residual_widget_render = gui.ImageWidget()
        # segment in source
        self.segment_widget = gui.ImageWidget()

        self.residual_layout = gui.Horiz(0)
        self.residual_layout.add_child(self.trg_image_widget)
        self.residual_layout.add_child(self.residual_widget_render)
        self.residual_layout.add_child(self.residual_widget)
        self.residual_layout.add_child(self.residual_widget_clicker)
        # self.residual_layout.add_child(self.gt_residual_widget)

        self.window_residual.add_child(self.residual_layout)


        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(left=spacing, top=vspacing, right=spacing, bottom=vspacing)

        self.ctrl_panel = gui.Vert(spacing, margins)

        ## Application control

        # Resume/pause
        resume_button = gui.ToggleSwitch("Resume/Pause")
        resume_button.set_on_clicked(self._on_pause_switch)
        resume_button.is_on = not self.config['paused']

        # Point cloud viz
        gt_button = gui.ToggleSwitch("Visualise GT SRC Point Cloud")
        gt_button.set_on_clicked(self._on_gt_switch)
        gt_button.is_on = False

        gt_button_trg = gui.ToggleSwitch("Visualise GT TRG Point Cloud")
        gt_button_trg.set_on_clicked(self._on_gt_switch_trg)
        gt_button_trg.is_on = False

        # Point cloud control
        self.pcd_vis_lv = gui.ListView()
        cloud_viz_options = ['source_colour', 'target_colour', 'mask', 'residual']
        self.pcd_vis_lv.set_items(cloud_viz_options)
        self.pcd_vis_lv.selected_index = self.pcd_vis_lv.selected_index + 1  # initially is -1, so now 1
        self.pcd_vis_lv.set_max_visible_items(4)
        self.pcd_vis_lv.set_on_selection_changed(self._on_list)
        self.pcd_vis_mode = cloud_viz_options[self.pcd_vis_lv.selected_index]

        # supporting_selector = gui.Combobox()
        # supporting_selector.add_item("trg")
        # supporting_selector.add_item("supp1")
        # supporting_selector.add_item("supp2")
        # supporting_selector.add_item("supp3")
        # supporting_selector.set_on_selection_changed(self._on_combo)

        ### Add panel children
        self.ctrl_panel.add_child(resume_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.pcd_vis_lv)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(gt_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(gt_button_trg)
        self.ctrl_panel.add_fixed(vspacing)
        # self.ctrl_panel.add_child(supporting_selector)
        self.ctrl_panel.add_fixed(vspacing)


        self.widget3d = gui.SceneWidget()


        self.window.add_child(self.ctrl_panel)
        self.window.add_child(self.widget3d)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        self.window.set_on_layout(self._on_layout)

        self.residual_widget_clicker.set_on_mouse(self._on_mouse_residual_choose)

        self.window.set_on_close(self._on_close)

        # Application variables
        self.is_running = resume_button.is_on
        self.is_done = False
        self.advance_one_frame = False

        # Visualization variables
        self.update_keyframe_render_viz = False
        self.normalize_est_depth = True

        # Point cloud mat
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.point_size = 3.0
        self.pcd_mat.shader = 'defaultUnlit'

        # Line mat
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.0
        self.line_mat.transmission = 1.0

        self.idx = 0

        self.scale = 1.0
        self.base_pose = torch.eye(4)

        random_pcd = o3d.geometry.PointCloud()
        random_pcd.points = o3d.utility.Vector3dVector(np.random.randn(100000,3))
        random_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100000,3))
        self.widget3d.scene.add_geometry("dense_estimation_points",random_pcd, self.pcd_mat)
        self.widget3d.scene.remove_geometry("dense_estimation_points")

        # Start processes
        torch.multiprocessing.set_start_method("spawn")

        self.setup_slam_processes()

        dataset = data.load_dataset(self.config)
        self.dataset = dataset 
        
        self.src = dataset[self.sfm.src_id]
        self.trg = dataset[self.sfm.trg_id]
        self.pose_gt = np.linalg.inv(self.trg['T']) @ self.src['T'] 

        self.src_gt_pcd = viz.depth_image_lift(self.src['depth'],
                                               self.src['intrinsics'],
                                               self.src['image'],
                                               T=self.src['T'])
                                               
        self.trg_gt_pcd = viz.depth_image_lift(self.trg['depth'],
                                               self.trg['intrinsics'],
                                               self.trg['image'],
                                               T=self.trg['T'])
         
    
        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()
        # self.update_main()
        self.vis_on = False
        self.segment_id = 0
        self.monocular_rescale = self.config['vis']['mono_align']


    def setup_slam_processes(self):
        self.waitev = torch.multiprocessing.Event()
        self.sfm = SfM(self.config_path, self.waitev)

        viz_queue = TupleTensorQueue(self.viz_device, self.viz_dtype) # Only want recent
        pause_queue = mp.Queue()

        kf_queue = TupleTensorQueue(self.viz_device, self.viz_dtype, maxsize=1)
        self.sfm.kf_queue = kf_queue

        self.sfm.viz_queue = viz_queue
        self.sfm.pause_queue = pause_queue

        self.viz_queue = viz_queue
        self.pause_queue = pause_queue

        self.kf_queue = kf_queue

    def start_slam_processes(self):

        self.sfm_done = False

        print("Starting sfm process...")
        # self.dataloader.start()
        self.sfm.start()
        print("Done.")

    def shutdown_slam_processes(self):
        self.waitev.set()
        print("Joining sfm,...")
        self.sfm.join()
        print("Done.")

    # def _on_combo(self, new_val, new_idx):
    #     # self.residual_frame_id = new_idx
    #     # post to renderer
    #     new_residual_img = viz.visualise_residual(self.curr_src_keyframe,
    #                                                 self.curr_supp_keyframes[new_idx], 
    #                                                 residuals=self.curr_residuals[new_idx],
    #                                                 segment_id=self.segment_id,
    #                                                 silent=True)
        
    #     if new_idx == 0:
    #         new_target_image = self.image_to_o3d(self.trg['image'].copy())
    #     else:
    #         frame_id = self.sfm.supp_ids[new_idx-1]
    #         new_target_image = self.image_to_o3d(self.dataset[frame_id]['image'].copy())


    #     def update_clicker_image():
    #         self.trg_image_widget.update_image(new_target_image)
    #         self.residual_widget.update_image(self.image_to_o3d(new_residual_img))

    #     def update_residual_img_id():
    #         self.residual_frame_id = new_idx

    #     gui.Application.instance.post_to_main_thread(
    #         self.window_residual, update_clicker_image)
    #     gui.Application.instance.post_to_main_thread(
    #         self.window_residual, update_residual_img_id)
    
    #     return

    def _on_list(self, new_val, is_dbl_click):
        self.pcd_vis_mode = new_val

    def _on_layout(self, ctx):
        em = ctx.theme.font_size
        panel_width = 20 * em
        rect = self.window.content_rect

        self.ctrl_panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.ctrl_panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                      rect.get_right() - x, rect.height)


    def _on_layout_residual(self, ctx):
        pass

    # Toggle callback: application's main controller
    def _on_pause_switch(self, is_on):
        self.is_running = is_on
        self.pause_queue.put(not is_on)

    def _on_mouse_residual_choose(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            x = event.x - self.residual_widget_clicker.frame.x
            y = event.y - self.residual_widget_clicker.frame.y
            
            x = x / self.residual_widget_clicker.frame.width * self.trg['image'].shape[1]
            y = y / self.residual_widget_clicker.frame.height * self.trg['image'].shape[0]


            dists = np.sum((etc.to_np(self.src_keypoints) - np.array([x, y]))**2, axis=-1)
            segment_id = np.argmin(dists)

            new_image = self._render_src_keypoints(segment_id)

            new_residual_img = None
            if self.curr_residuals is not None:
                new_residual_img = viz.visualise_residual(self.curr_src_keyframe,
                                                          self.curr_supp_keyframes[self.residual_frame_id], 
                                                          residuals=self.curr_residuals[self.residual_frame_id],
                                                          segment_id=segment_id,
                                                          silent=True)

            # This is not called on the main thread, so we need to
            # post to the main thread to safely access UI items.
            def update_segment_id():
                self.segment_id = segment_id

            def update_clicker_image():
                if new_residual_img is not None:
                    self.residual_widget.update_image(self.image_to_o3d(new_residual_img))
                self.residual_widget_clicker.update_image(self.image_to_o3d(new_image))

            gui.Application.instance.post_to_main_thread(
                self.window_residual, update_segment_id)
            
            gui.Application.instance.post_to_main_thread(
                self.window_residual, update_clicker_image)

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _render_src_keypoints(self, segment_id):
        new_image = viz.scatter_keypoints(self.src['image'],
                                          self.src_keypoints,
                                          selected_id=segment_id)
        
        return new_image
    
    def _on_gt_switch(self, is_on):
        self.show_gt_pcd = is_on

        def update_gt_pcd():
            if self.show_gt_pcd:
                self.widget3d.scene.add_geometry("gt_points", self.src_gt_pcd, self.pcd_mat)
            else:
                self.widget3d.scene.remove_geometry("gt_points")

        gui.Application.instance.post_to_main_thread(
            self.window, update_gt_pcd)
        
        return 
    
    def _on_gt_switch_trg(self, is_on):
        self.show_gt_pcd_trg = is_on

        def update_gt_pcd():
            if self.show_gt_pcd_trg:
                self.widget3d.scene.add_geometry("gt_points_trg", self.trg_gt_pcd, self.pcd_mat)
            else:
                self.widget3d.scene.remove_geometry("gt_points_trg")

        gui.Application.instance.post_to_main_thread(
            self.window, update_gt_pcd)
        
        return 


    def _on_start(self):
        pass

    def _on_close(self):
        self.is_done = True
        return True

    def setup_camera_view(self, pose, base_pose):
        center, eye, up = o3d_tools.pose_to_camera_setup(pose, base_pose, self.scale)
        self.widget3d.look_at(center, eye, up)


    def init_render_residual(self):
        img = 0.57 * np.ones((self.trg['image'].shape[0], self.trg['image'].shape[1], 3))
        img_v2 = 0.76 * np.ones((self.trg['image'].shape[0], self.trg['image'].shape[1], 3))
        img_v3 = 0.78 * np.ones((self.trg['image'].shape[0], self.trg['image'].shape[1], 3))

        img = (img * 255).astype(np.uint8)
        img_v2 = (img_v2 * 255).astype(np.uint8)
        img_v3 = (img_v3 * 255).astype(np.uint8)

        self.trg_image_widget.update_image(self.image_to_o3d(self.trg['image'].copy()))

        self.residual_widget.update_image(self.image_to_o3d(img))
        self.residual_widget_clicker.update_image(self.image_to_o3d(img_v2))
        self.residual_widget_render.update_image(self.image_to_o3d(img_v3))
    
    def init_render(self):
        self.window.set_needs_layout()

        fov = 60.0
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(fov, bounds, bounds.get_center())


        self.setup_camera_view(self.src['T'], self.base_pose)

    def update_idx_text(self):
        self.idx_label.text = 'Idx: {:8d}'.format(self.idx)

    def update_curr_residual_render(self, render_residual, render_pcd):
        self.residual_widget.update_image(render_residual)
        self.residual_widget_render.update_image(render_pcd)


    def image_to_o3d(self, image):
        rgb_np_uint8 = np.asarray(image)
        rgb_np_uint8 = np.ascontiguousarray(rgb_np_uint8)
        rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
        return rgb_img.to_legacy()
    
    
    def get_dense_pcd(self, residuals, src_to_trg_pose, scale=1):
        pts = residuals['src_pts']

        def residual_to_heat(r):
            r = torch.abs(r).mean(dim=1, keepdims=True)
            red = torch.tanh(r)
            red = torch.cat([red, torch.zeros_like(red), torch.zeros_like(red)], dim=1)

            return red

        colour_map = {'source_colour': residuals['src_pixels'],
                      'target_colour': residuals['src_in_trg_pixels'],
                      'mask': residuals['full_mask'].float().expand(-1, 3,-1),
                      'residual': residual_to_heat(residuals['residual_raw'])
                      }
        
        pts_colour = colour_map[self.pcd_vis_mode]
        each = self.config['vis']['pts_show_every']

        points = (etc.to_np(scale * pts[::each])).copy()
        points_colour = (etc.to_np(pts_colour[0][:3, ::each].T))

        if points_colour.shape[1] == 1:
            points_colour = np.tile(points_colour, (1, 3))

        curr_dense_pcd = o3d.geometry.PointCloud()
        curr_dense_pcd.points = o3d.utility.Vector3dVector(points)
        curr_dense_pcd.colors = o3d.utility.Vector3dVector(points_colour)

        
        render_pts = (etc.to_np(transform_points(scale * pts, src_to_trg_pose))).copy()

        pts_to_render = residuals['src_pixels'][0, :3]
        render_colour = pts_to_render.T 


        
        H, W = self.trg['image'].shape[:2]
        image_rendered = render_pcd(render_pts, render_colour, instrinsic_scaled_K(self.trg['intrinsics'], 1/4.0), H // 4, W // 4)

        image_rendered = cv2.resize(image_rendered, (W, H), interpolation=cv2.INTER_LINEAR)
        curr_dense_pcd.transform(self.src['T'])
        return curr_dense_pcd, image_rendered
    
    def get_frustum(self, pose):
        intr = self.src['intrinsics']
        img_size = self.src['image'].shape[:2]


        frustum = frustum_lineset(intr,
                                  img_size,
                                  pose, scale=0.2)
        return frustum
    

    def update_keyframe_render(self, dense_pcd, current_pose): 
        self.widget3d.scene.remove_geometry("dense_estimation_points")
        self.widget3d.scene.add_geometry("dense_estimation_points", dense_pcd, self.pcd_mat)
    

    def update_pose_render(self, src_gt_frustum, pred_supp_frustum, gt_supp_frustum):
        self.widget3d.scene.remove_geometry("T_src_gt")
        self.widget3d.scene.add_geometry("T_src_gt", src_gt_frustum, self.line_mat)

        for i in range(len(pred_supp_frustum)):
            self.widget3d.scene.remove_geometry("T_trg_" + str(i))
            self.widget3d.scene.remove_geometry("T_trg_gt_" + str(i))

            self.widget3d.scene.add_geometry("T_trg_" + str(i), pred_supp_frustum[i], self.line_mat)
            self.widget3d.scene.add_geometry("T_trg_gt_" + str(i), gt_supp_frustum[i], self.line_mat)
        
    def setup_camera_view(self, pose, base_pose):
        center, eye, up = o3d_tools.pose_to_camera_setup(pose, base_pose, self.scale)
        self.widget3d.look_at(center, eye, up)


    def get_vis_poses(self, delta_poses):
        world_poses = []
        dscaled_poses = []
        frustums = []

        if self.monocular_rescale:
                # assuming first pose is in the right correspondance 
                _, scaling_factor = renorm_translation(etc.to_np(delta_poses[0]), 
                                                                 self.scale_gt)
                print('scaling factor inferred', scaling_factor)
        else:
            scaling_factor = 1.0

        for pos in delta_poses:
            new_pose = apply_scale(etc.to_np(pos), scaling_factor)

            new_pose_trg = self.src['T'] @ np.linalg.inv(new_pose)
            new_pose_frustum = self.get_frustum(new_pose_trg) 
            
            dscaled_poses.append(new_pose)
            world_poses.append(new_pose_trg)
            frustums.append(new_pose_frustum)
        
        return frustums, world_poses, dscaled_poses, scaling_factor
    
    def get_gt_poses(self, frame_ids):
        frustums = []
        gt_poses = []
        for frame_id in frame_ids:
            pose = self.dataset[frame_id]['T']     
            trg_gt_frustum = self.get_frustum(pose)

            trg_gt_frustum.paint_uniform_color([1,0,0])
            
            frustums.append(trg_gt_frustum)
            gt_poses.append(pose)

        return frustums, gt_poses

    def update_main(self):

        # Initialize processes
        self.start_slam_processes()

        # Initialize rendering
        gui.Application.instance.post_to_main_thread(
            self.window, self._on_start)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render())
        gui.Application.instance.post_to_main_thread(
            self.window_residual, lambda: self.init_render_residual())
        # Record data
        self.timestamps = []
        self.est_poses = np.array([]).reshape(0,4,4)

        # Rendering helper functions

        def update_keyframe_render_helper():
            self.update_keyframe_render_viz = False
            self.update_keyframe_render(curr_dense_pcd, current_pose)
            return

        def update_pose_render_helper():
            self.update_pose_render(src_gt_frustum, supp_frustums, supp_gt_frustums)
            return
        # Main loop


        src_kf = self.kf_queue.pop(block=True, timeout=None)
        src_kf = src_kf[0]
        spatial_dims = self.src['image'].shape[:2]
        self.src_keypoints = etc.to_np(point_utils.denormalise_coordinates(src_kf.keypoints, spatial_dims).flip(-1))
        
        self.scale_gt = get_translation_norm(self.pose_gt)

        def update_clicker():
            new_image = self._render_src_keypoints(self.segment_id)
            self.residual_widget_clicker.update_image(self.image_to_o3d(new_image))

        gui.Application.instance.post_to_main_thread(
            self.window_residual, update_clicker)
        # self.src_key

        self.residual_frame_id = 0

        while not self.sfm_done:
            # Receive keyframe visualization
            viz_data = self.viz_queue.pop_until_latest(block=False, timeout=0.01)
            if viz_data is not None:
                if viz_data[0] == "end":
                    self.sfm_done = True
                else:
                    residuals, current_pose, src_keyframe, supp_keyframes = viz_data

                    # infer world poses scaled to gt
                    supp_frustums, supp_poses, scaled_poses, scaling_factor = self.get_vis_poses(current_pose)

                    self.curr_residuals = residuals
                    self.curr_src_keyframe = src_keyframe
                    self.curr_supp_keyframes = supp_keyframes

                    print('residual frame id', self.residual_frame_id)

                    supp_gt_frustums, supp_gt_poses = self.get_gt_poses([self.sfm.trg_id] + self.sfm.supp_ids)

                    src_gt_frustum = self.get_frustum(self.src['T'])
                    src_gt_frustum.paint_uniform_color([0,1,0])

                    curr_dense_pcd, render_img = self.get_dense_pcd(residuals[self.residual_frame_id], 
                                                                    scaled_poses[self.residual_frame_id],
                                                                    scaling_factor)
                    

                    residual_img = viz.visualise_residual(src_keyframe,
                                                          supp_keyframes[self.residual_frame_id], 
                                                          residuals=residuals[self.residual_frame_id],
                                                          segment_id=self.segment_id,
                                                          silent=True)


                    residual_img = self.image_to_o3d(residual_img)
                    render_img = self.image_to_o3d(render_img)


                    gui.Application.instance.post_to_main_thread(
                        self.window, update_keyframe_render_helper)
                    
                    gui.Application.instance.post_to_main_thread(
                        self.window, update_pose_render_helper) 

                    gui.Application.instance.post_to_main_thread(
                        self.window_residual, lambda: self.update_curr_residual_render(residual_img,
                                                                                       render_img))
                               
            release_data(viz_data)

            self.idx += 1

        self.shutdown_slam_processes()
