import torch # Must import before Open3D when using CUDA!
import torch.multiprocessing as mp

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

import threading
import data
import yaml
import copy 


from tool import viz
from tool.multiprocess import TupleTensorQueue, release_data

import torch.utils.dlpack
import tool.o3d_tools as o3d_tools
from tool.o3d_frustum import create_frustum, Frustum

import tool.point_utils as point_utils
from tool.pose_utils import get_sorted_by_timestamp
from tool.pose_utils import transfer_scale, apply_scale
import tool.etc as etc
from tool.etc import to_np

from odometery.odometery import Odometery

def o3t_from_t(t):
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(t))

def enable_widget(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

def transform_points(pts, T):
    pts = np.einsum('ij, nj -> ni', T[:3, :3], pts)  + T[:3,3]
    return pts 


def cat_o3d_pointclouds(pcds):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points) for pcd in pcds], axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.colors) for pcd in pcds], axis=0))
    return pcd


class OdomWindow():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        self.viz_device = torch.device("cuda:0")
        self.viz_dtype = torch.float32


        self.window = gui.Application.instance.create_window(
            'Monocular VO', width=1920, height=1080) # 1280 x 720 for recording
        em = 10

        self.window_residual = gui.Application.instance.create_window(
            'Residual Reprojected', width=1024 // 3 * 2, height=768 // 3)
        
        self.trg_image_widget = gui.ImageWidget()
        self.residual_widget = gui.ImageWidget()
        self.residual_widget_clicker = gui.ImageWidget() 
        self.residual_widget_render = gui.ImageWidget()
        # segment in source
        self.segment_widget = gui.ImageWidget()
        # gt residual
        self.gt_residual_widget = gui.ImageWidget()

        self.residual_layout = gui.Horiz(0)
        
        self.residual_layout.add_child(self.residual_widget)
        self.residual_layout.add_child(self.residual_widget_clicker)

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
        self.gt_button = gui.ToggleSwitch("Visualise GT Point Cloud")
        self.gt_button.set_on_clicked(self._on_gt_switch)
        self.gt_button.is_on = False
        self.show_gt_pcd = False


        self.src_frame_slider = gui.Slider(gui.Slider.INT)
        self.src_frame_slider.set_limits(0, 0)
        self.src_frame_slider.int_value = int(0)
        self.src_frame_slider.set_on_value_changed(self._on_src_frame_slider)
        self.trg_frame_slider = gui.Slider(gui.Slider.INT)
        self.trg_frame_slider.set_limits(0, 0)
        self.trg_frame_slider.int_value = int(0)
        self.trg_frame_slider.set_on_value_changed(self._on_trg_frame_slider)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        # ### Data tab
        tab_data = gui.Vert(0, tab_margins)
        self.curr_image_tab = gui.ImageWidget()
        tab_data.add_child(self.curr_image_tab)

        self.cameras_chbox = gui.Checkbox("Show cameras")
        self.cameras_chbox.set_on_checked(self._on_cam_checkbox)     
        self.cameras_chbox.checked = True

        self.follow_kf_chbox = gui.Checkbox("Follow KF")
        self.follow_kf_chbox.set_on_checked(self._on_follow_checkbox)     
        self.follow_kf_chbox.checked = False

        self.follow_menu = gui.Combobox()
        self.follow_menu.add_item("Free")
        self.follow_menu.add_item("Follow KF pose")
        self.follow_menu.add_item("Follow Track pose")
        self.follow_menu.set_on_selection_changed(self._on_follow_menu)

        self.follow_kf_geom_chbox = gui.Checkbox("Follow KF Geometry")
        self.follow_kf_geom_chbox.set_on_checked(self._on_follow_checkbox_geom)     
        self.follow_kf_geom_chbox.checked = True
        
        self.show_all_geoms_chbox = gui.Checkbox("Show Full Geometry")
        self.show_all_geoms_chbox.set_on_checked(self._on_show_all_geoms)
        self.show_all_geoms_chbox.checked = False


        self.show_gt_traj_chbox = gui.Checkbox("Show GT Trajectory")
        self.show_gt_traj_chbox.set_on_checked(self._on_show_gt_traj)
        self.show_gt_traj_chbox.checked = False
        ### Add panel children
        self.ctrl_panel.add_child(resume_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.gt_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(gui.Label("Source KF Selector"))
        self.ctrl_panel.add_child(self.src_frame_slider)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(gui.Label("Target KF Selector"))
        self.ctrl_panel.add_child(self.trg_frame_slider)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.cameras_chbox)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.follow_menu)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.follow_kf_geom_chbox)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.show_all_geoms_chbox)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.show_gt_traj_chbox)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(tab_data)

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

        # Start processes
        torch.multiprocessing.set_start_method("spawn")

        self.setup_slam_processes()

        dataset = data.load_dataset(self.config)
        self.dataset = dataset 
        
        self.trg = dataset[0]
        self.img_shape = self.trg['image'].shape
        self.intriniscs = self.trg['intrinsics']

        self.colour_map = {'pred': [0.0, 0.0, 1.0],
                           'gt': [1.0, 0.0, 0.0],
                           'supp': [0.0, 0.0, 0.0]}
        
        self.frustum_dict = {}
        self.residuals = None
        self.prev_src_ts = None

        self.vis_on = False
        self.segment_id = 0
        self.monocular_rescale = True
        if 'mono_align' in self.config['vis']:
            self.monocular_rescale = self.config['vis']['mono_align']
        

        self.current_frustums = []
        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()


    def setup_slam_processes(self):
        self.waitev = torch.multiprocessing.Event()
        self.sfm = Odometery(self.config_path, self.waitev)

        viz_queue = TupleTensorQueue(self.viz_device, self.viz_dtype) # Only want recent
        pause_queue = mp.Queue()

        kf_queue = TupleTensorQueue(self.viz_device, self.viz_dtype, maxsize=1)
        self.sfm.kf_queue = kf_queue

        self.sfm.viz_queue = viz_queue
        self.sfm.pause_queue = pause_queue

        self.viz_queue = viz_queue
        self.pause_queue = pause_queue

        self.kf_queue = kf_queue

        self.frame_queue = mp.Queue()
        self.sfm.frame_queue = self.frame_queue

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
        o3d.visualization.gui.Application.instance.quit()
        print("killing visualization process")

    def _on_follow_menu(self, new_val, new_idx):
        pass

    def _on_cam_checkbox(self, val):
        pass

    def _on_follow_checkbox(self, val):
        pass

    def _on_follow_checkbox_geom(self, val):
        pass

    def _on_show_gt_traj(self, val):
        for name in self.frustum_dict.keys():
            if name.startswith('T_gt'):
                self.widget3d.scene.show_geometry(name, val)
        return

    def _on_show_all_geoms(self, val):
        
        def update_helper():
            if val:
                self.update_pcd(self.cat_dense_pcd)
                if self.show_gt_pcd:
                    self.update_gt_pcd(self.cat_dense_gts)
            else:
                src_id = self.src_frame_slider.int_value
                self.update_pcd(self.dense_pcds[src_id])
                
                if self.show_gt_pcd:
                    self.update_gt_pcd(self.gt_pcds[src_id])
        pass

        # post
        gui.Application.instance.post_to_main_thread(
            self.window, update_helper)
        return


    def smart_residual_render(self, src_id, trg_id):
        trg_kf = None
        trg_timestamp = self.timestamps[trg_id]
        
        trg_kf = self.supp_kfs[src_id][trg_id]
        new_residual_img = viz.visualise_residual_batch_v2(self.kfs[src_id],
                                     trg_kf, 
                                     residuals=self.residuals[src_id],
                                     residual_id=trg_id,
                                     segment_id=self.segment_id,
                                     silent=True)  
 
        return new_residual_img, trg_timestamp    

    def _on_src_frame_slider(self, val):
        val = int(val)

        def update_trg_slider():
            self.trg_frame_slider.set_limits(0,  len(self.supp_kfs[val]) - 1)
        
        gui.Application.instance.post_to_main_thread(
            self.window, update_trg_slider)

        def update():
            if self.show_all_geoms_chbox.checked:
                return
            self.update_pcd(self.dense_pcds[val])
            if self.show_gt_pcd:
                self.update_gt_pcd(self.gt_pcds[val])
            return

        gui.Application.instance.post_to_main_thread(
            self.window, update)
        
        return


    def _on_trg_frame_slider(self, val):
        val = int(val)
        return

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
    
    def add_camera(self, cpose, name, color=[0,1,0]):
        C2W = cpose
        # If there is no frustum registered, create one
        # self.cameras_chbox.checked = True
        if not name in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color)
            # self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.line_mat)

        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)

        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)

        return frustum


    # Toggle callback: application's main controller
    def _on_pause_switch(self, is_on):
        self.is_running = is_on
        self.pause_queue.put(not is_on)

    def _on_mouse_residual_choose(self, event):

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:

            x = event.x - self.residual_widget_clicker.frame.x
            y = event.y - self.residual_widget_clicker.frame.y
            

            x = x / self.residual_widget_clicker.frame.width * self.img_shape[1]
            y = y / self.residual_widget_clicker.frame.height * self.img_shape[0]


            dists = np.sum((etc.to_np(self.src_keypoints) - np.array([x, y]))**2, axis=-1)
            segment_id = np.argmin(dists)


            # This is not called on the main thread, so we need to
            # post to the main thread to safely access UI items.
            def update_segment_id():
                self.segment_id = segment_id

            gui.Application.instance.post_to_main_thread(
                self.window_residual, update_segment_id)
            # self.widget3d.scene.scene.render_to_depth_image(depth_callback)
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
            src_id = self.src_frame_slider.int_value
            if self.show_gt_pcd and self.gt_pcds is not None:
                self.widget3d.scene.add_geometry("gt_points", self.gt_pcds[src_id], self.pcd_mat)
            else:
                self.widget3d.scene.remove_geometry("gt_points")

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
        img = 0.57 * np.ones((self.img_shape[0], self.img_shape[1], 3))
        img_v2 = 0.76 * np.ones((self.img_shape[0], self.img_shape[1], 3))

        img = (img * 255).astype(np.uint8)
        img_v2 = (img_v2 * 255).astype(np.uint8)

        self.residual_widget.update_image(self.image_to_o3d(img))
        self.residual_widget_clicker.update_image(self.image_to_o3d(img_v2))
    
    def init_render(self):
        self.window.set_needs_layout()

        fov = 60.0
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(fov, bounds, bounds.get_center())

        self.setup_camera_view(np.eye(4), self.base_pose)


    def update_curr_residual_render(self, render_residual):
        self.residual_widget.update_image(render_residual)


    def image_to_o3d(self, image):
        rgb_np_uint8 = np.asarray(image)
        rgb_np_uint8 = np.ascontiguousarray(rgb_np_uint8)
        rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
        return rgb_img.to_legacy()
    
    def update_residual_supp(self,  kfs, residuals, poses, 
                             timestamps, connectivity,
                             supp_kfs, align_info):
        kf_id = len(self.kfs) - 1

        src_id = self.src_frame_slider.int_value

        last_pose = etc.to_np(poses[kf_id])
        dense_pcd = self.get_dense_pcd(residuals[-1], last_pose, align_info)
        self.dense_pcds[-1] = dense_pcd 

        dataset_sample = self.dataset[int(timestamps[kf_id])]
        gt_dense_pcd = viz.depth_image_lift(dataset_sample['depth'],
                                            dataset_sample['intrinsics'],
                                            dataset_sample['image'],
                                            T=dataset_sample['T'])
        self.gt_pcds[-1] = gt_dense_pcd

        self.cat_dense_pcd = cat_o3d_pointclouds(self.dense_pcds)
        self.cat_dense_gts = cat_o3d_pointclouds(self.gt_pcds)
        

        def update_pcd_helper():
            # resent clicker since we've got a new frame 
            kf_id = len(self.kfs) - 1
            if timestamps[kf_id] != self.timestamps[kf_id]:
                print('dropped supp update!')
                return 
            if self.show_all_geoms_chbox.checked:
                self.update_pcd(self.cat_dense_pcd)
                if self.show_gt_pcd:
                    self.update_gt_pcd(self.cat_dense_gts)
            else:
                self.update_pcd(dense_pcd)
                if self.show_gt_pcd:
                    self.update_gt_pcd(self.gt_pcds[src_id])
            return

        gui.Application.instance.post_to_main_thread(
            self.window, update_pcd_helper)
        
    
    def set_current_residual(self, kfs, residuals, poses, 
                             timestamps, connectivity,
                             supp_kfs, align_info):
        num_map_kfs = len(residuals)
        self.kfs = kfs
        self.supp_kfs = supp_kfs

        src_kf_changed = False


        if self.follow_kf_geom_chbox.checked:
            src_id = num_map_kfs - 1
        else:
            src_id = min(self.src_frame_slider.int_value, num_map_kfs - 1)

        if self.prev_src_ts is not None and self.prev_src_ts == timestamps[src_id]:
            src_kf_changed = False
        else:
            src_kf_changed = True    

        self.prev_src_ts = timestamps[src_id]
        
        if src_kf_changed:
            trg_id = len(self.supp_kfs[src_id]) - 1
        else:
            trg_id = min(self.trg_frame_slider.int_value, len(self.supp_kfs[src_id]) - 1)

        def update_sliders():
            self.src_frame_slider.set_limits(0, num_map_kfs-1)
            self.src_frame_slider.int_value = (src_id)
            self.trg_frame_slider.set_limits(0, len(self.supp_kfs[src_id]) - 1)
            self.trg_frame_slider.int_value = (trg_id)
        
        gui.Application.instance.post_to_main_thread(
            self.window, update_sliders)

        start_pose = self.base_pose

        dense_pcds = []
        gt_pcds = []
        ## pre calculate dense pcds
        for kf_id in range(num_map_kfs):
            last_pose = etc.to_np(poses[kf_id])
            dense_pcd = self.get_dense_pcd(residuals[kf_id], last_pose, align_info)
            dense_pcds.append(dense_pcd)

            dataset_sample = self.dataset[int(timestamps[kf_id])]
            gt_dense_pcd = viz.depth_image_lift(dataset_sample['depth'],
                                                dataset_sample['intrinsics'],
                                                dataset_sample['image'],
                                                T=dataset_sample['T'])
            gt_pcds.append(gt_dense_pcd)
        
        
        dense_pcd = dense_pcds[src_id]

        cat_dense_pcd = cat_o3d_pointclouds(dense_pcds)
        cat_dense_gts = cat_o3d_pointclouds(gt_pcds)

        self.dense_pcds = dense_pcds
        self.gt_pcds = gt_pcds

        self.cat_dense_pcd = cat_dense_pcd
        self.cat_dense_gts = cat_dense_gts

        self.residuals = residuals
        
        self.connectivity = connectivity
        self.align_info = align_info
        self.poses = poses
        self.timestamps = timestamps
        self.set_src_keyframe(kfs[src_id], timestamps[src_id])


        new_clicker = self._render_src_keypoints(self.segment_id)
        new_clicker = self.image_to_o3d(new_clicker)


        residual_img, _ = self.smart_residual_render(src_id, trg_id)
        residual_img = self.image_to_o3d(residual_img)

 
        def update_pcd_helper():
            # resent clicker since we've got a new frame 
            # self.update_keyframe_render_viz = False
            if self.show_all_geoms_chbox.checked:
                self.update_pcd(cat_dense_pcd)
                if self.show_gt_pcd:
                    self.update_gt_pcd(cat_dense_gts)
            else:
                self.update_pcd(dense_pcd)
                if self.show_gt_pcd:
                    self.update_gt_pcd(gt_pcds[src_id])
            return
        
        def update_residual_helper():
            if src_kf_changed:
                self.segment_id = 0
            self.set_src_keyframe(kfs[src_id], timestamps[src_id])
            self.residual_widget.update_image(residual_img)
            self.residual_widget_clicker.update_image(new_clicker)
        

        gui.Application.instance.post_to_main_thread(
            self.window, update_pcd_helper)
        
    
        gui.Application.instance.post_to_main_thread(
            self.window_residual, update_residual_helper)
        
        return
    
    def update_clicker_image(self):
        if self.residuals is None:
            return 
        src_id = self.src_frame_slider.int_value
        self.set_src_keyframe(self.kfs[src_id], self.timestamps[src_id])

        new_image = self._render_src_keypoints(self.segment_id)
        src_id = self.src_frame_slider.int_value
        trg_id = self.trg_frame_slider.int_value

        if self.residuals is not None:
            new_residual_img, _ = self.smart_residual_render(src_id, trg_id)
    
        def sender_to_residual():
            self.residual_widget.update_image(self.image_to_o3d(new_residual_img))
            self.residual_widget_clicker.update_image(self.image_to_o3d(new_image))

        gui.Application.instance.post_to_main_thread(
            self.window_residual, sender_to_residual)
        return

    
    def get_dense_pcd(self, residuals, T_WC, align_info=None):
        scale = 1.0
        pts = residuals['src_pts']
        pts_colour = residuals['src_pixels']
        each = self.config['vis']['pts_show_every']

        points = (etc.to_np(scale * pts[::each])).copy()
        points_colour = (etc.to_np(pts_colour[0][:3, ::each].T))

        curr_dense_pcd = o3d.geometry.PointCloud()
        curr_dense_pcd.points = o3d.utility.Vector3dVector(points)
        curr_dense_pcd.colors = o3d.utility.Vector3dVector(points_colour)

        if align_info is not None:
            scale = align_info['s']
            curr_dense_pcd.scale(scale, center=(0,0,0))

            T_WC = apply_scale(T_WC, align_info)

        curr_dense_pcd.transform(T_WC)
            
        return curr_dense_pcd 
    
    def update_curr_tab(self, frame_id):
        frame_id = int(frame_id)
        new_target_image = self.image_to_o3d(self.dataset[frame_id]['image'].copy())


        def update_clicker_trg_image():
            self.curr_image_tab.update_image(new_target_image)

        gui.Application.instance.post_to_main_thread(
            self.window, update_clicker_trg_image)
        return
    
    def set_src_keyframe(self, src_kf, timestamp):
        self.src = self.dataset[int(timestamp)]
        spatial_dims = self.img_shape[:2]
        self.src_keypoints = etc.to_np(point_utils.denormalise_coordinates(src_kf['keypoints'], 
                                                                                 spatial_dims).flip(-1))
        self.curr_src_keyframe = src_kf

    def update_pcd(self, dense_pcd): 
        self.widget3d.scene.remove_geometry("dense_estimation_points")
        self.widget3d.scene.add_geometry("dense_estimation_points", dense_pcd, self.pcd_mat)
    
    def update_gt_pcd(self, gt_dense_pcd): 
        self.widget3d.scene.remove_geometry("gt_points")
        self.widget3d.scene.add_geometry("gt_points", gt_dense_pcd, self.pcd_mat)

    def set_view_pose(self, frust):
        center, eye_behind, up, latest_pose = frust.view_dir_behind

        self.setup_camera_view(latest_pose, self.base_pose)
        return 

    def update_pose_renders(self, gt_frustums, pred_frustums, timestamps, force_follow=False):
        assert(len(gt_frustums) == len(pred_frustums))
        assert(len(gt_frustums) == len(timestamps))

        for gt_frust, pred_frust, ts in zip(gt_frustums, pred_frustums, timestamps):
            self.add_camera(pred_frust, "T_" + ts, color=self.colour_map['pred'])
            if self.show_gt_traj_chbox.checked:
                self.add_camera(gt_frust, "T_gt_" + ts, color=self.colour_map['gt'])
        
        if self.follow_menu.selected_index == 1 or force_follow:
            print('Following latest keyframe')
            latest_frust = self.frustum_dict['T_' + timestamps[-1]]
            center, eye_behind, up, latest_pose = latest_frust.view_dir_behind


            self.setup_camera_view(latest_pose, self.base_pose)
        
        return
    
    def update_supp_pose_renders(self, supp_views, supp_ts, gt_supp_views=None, gt_supp_ts=None):
        for ts, pred_frust in zip(supp_ts, supp_views):
            assert(ts.startswith('supp'))
            self.add_camera(pred_frust, "T_supp_" + ts, color=self.colour_map['supp'])

        if gt_supp_views is not None and self.show_gt_traj_chbox.checked:
            for gt_frust, ts in zip(gt_supp_ts, gt_supp_views):
                assert(ts.startswith('supp'))
                self.add_camera(gt_frust, "T_gt_" + ts, color=self.colour_map['gt'])
        return 

    def update_pose_track(self, pred_track, gt_track=None):
        self.add_camera(pred_track, "T_track", color=[0, 1, 0])

        if self.follow_menu.selected_index == 2:
            latest_frust = self.frustum_dict['T_track']
            center, eye_behind, up, latest_pose = latest_frust.view_dir_behind
            self.setup_camera_view(latest_pose, self.base_pose)

        if gt_track is not None and self.show_gt_traj_chbox.checked:
            self.add_camera(gt_track, "T_gt_track", color=[1, 1, 0])
        
        return  
    
    def realign_poses(self, add_poses, add_ts):
        add_gt_poses = [self.dataset[int(t)]['T'] for t in add_ts]
        add_pred_poses = [to_np(pose) for pose in add_poses]


        pred_pose_dict = copy.deepcopy(self.pred_pose_dict)
        gt_pose_dict = copy.deepcopy(self.gt_pose_dict)
        
        align_info = None
        if self.monocular_rescale:
            pred_poses_viz, align_info = transfer_scale(get_sorted_by_timestamp(gt_pose_dict) ,
                                                        get_sorted_by_timestamp(pred_pose_dict),
                                                        anchor_rotation=True)
            gt_poses_viz, timestamps_viz = get_sorted_by_timestamp(gt_pose_dict, True)
            add_pred_poses = [apply_scale(pose, align_info) for pose in add_pred_poses]
        else:
            pred_poses_viz, timestamps_viz = get_sorted_by_timestamp(pred_pose_dict, True)
            gt_poses_viz, timestamps_viz = get_sorted_by_timestamp(gt_pose_dict, True)

        result = {'pred_poses': pred_poses_viz,
                  'gt_poses': gt_poses_viz,
                  'timestamps': timestamps_viz,
                  'add_pred_poses': add_pred_poses,
                  'add_gt_poses': add_gt_poses,
                  'add_timestamp': add_ts,
                  'align_info': align_info}
        
        return result

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


        def update_pose_helper():
            self.setup_camera_view(etc.to_np(gt_poses[-1]), start_pose)

        def update_pcd_helper():
            self.update_pcd(dense_pcd)
            return

        self.gt_pose_dict = {}
        self.pred_pose_dict = {}  
        self.track_pose_dict = {}

        start_pose = np.eye(4)
        pose_set = False
        align_info = None

        viz_data = self.viz_queue.pop(block=True, timeout=None)
        self.max_num_supps = 0


        self.current_id = self.config['dataset']['start_id']

        while not self.sfm_done:
            curr_frame_info = self.kf_queue.pop_until_latest(block=False, timeout=0.001)
            if curr_frame_info is not None:
                self.update_curr_tab(curr_frame_info[1])

            self.update_clicker_image()
            if viz_data is not None:
                if viz_data[0] == "end":
                    self.sfm_done = True
                    # self.save_trajectory()
                    continue
                
                if viz_data[0] == 'init':
                    print('init', flush=True)
                    _, kfs, timestamps, poses, residuals, traj_updates = viz_data

                    init_timestamps = timestamps
                    gt_poses = [self.dataset[int(t)]['T'] for t in timestamps]
                    pred_poses = [start_pose @ to_np(pose) for pose in poses]

                    for t, pose in zip(timestamps, pred_poses):
                        self.pred_pose_dict[t] = pose

                    for t, pose in zip(timestamps, gt_poses):
                        self.gt_pose_dict[t] = pose

                    if traj_updates is not None:
                        print('VIZ: resetting trajectory')
                        kf_traj, track_traj = traj_updates
                        print('Traj sizes', len(kf_traj), len(track_traj))
                        for t, pose in kf_traj.items():
                            self.pred_pose_dict[t] = pose
                            self.gt_pose_dict[t] = self.dataset[int(t)]['T']
                        for t, pose in track_traj.items():
                            self.track_pose_dict[t] = pose

                        rescaled_poses = self.realign_poses([], [])

                        align_info = rescaled_poses['align_info']

                        def update_reinit_pose_render_helper():
                            self.update_pose_renders(rescaled_poses['gt_poses'], 
                                                     rescaled_poses['pred_poses'], 
                                                     rescaled_poses['timestamps'])
                            return
                        
                        gui.Application.instance.post_to_main_thread(
                            self.window, update_reinit_pose_render_helper)
                        

                    gt_frustums = [pose for pose in gt_poses]
                    pred_frustums = [pose for pose in pred_poses]

                    src_kf_id = 0 
                    last_pose = start_pose @ etc.to_np(poses[src_kf_id])
                    dense_pcd = self.get_dense_pcd(residuals[src_kf_id], last_pose)
                    self.set_src_keyframe(kfs[src_kf_id], timestamps[src_kf_id])
                    del src_kf_id

                    def update_init_pose_render_helper():
                        self.update_pose_renders(gt_frustums, pred_frustums, init_timestamps,
                                                 force_follow=True)
                        return

                    gui.Application.instance.post_to_main_thread(
                        self.window, update_init_pose_render_helper)
                    
                    gui.Application.instance.post_to_main_thread(
                        self.window, update_pcd_helper)
                    

                elif viz_data[0] == 'tracking':
                    _, kfs, timestamps, poses, residuals, factors = viz_data
                    # print('Viz: tracking', flush=True)
                    
                    track_pose = start_pose @ etc.to_np(poses[0])

                    self.track_pose_dict[timestamps[-1]] = track_pose

                    if self.monocular_rescale and align_info is not None:
                        track_pose = apply_scale(track_pose, align_info)

                    tracking_pose = track_pose
                    gt_track_pose = self.dataset[int(timestamps[-1])]['T']


                    gui.Application.instance.post_to_main_thread(
                        self.window,  lambda: self.update_pose_track(tracking_pose, gt_track_pose))
                    
                elif viz_data[0] == 'supp_mapping':
                    _, all_kfs, timestamps, poses, residuals, factors = viz_data
                    assert(len(residuals) == 1)
                    
                    kfs, supp_kfs = all_kfs
                    # print('VIZ: supp mapping', flush=True)

                    self.update_residual_supp(kfs, residuals, poses, 
                                              timestamps, factors, 
                                              supp_kfs, align_info)
                elif viz_data[0] == 'mapping':
                    # mapping
                    _, all_kfs, timestamps, poses, residuals, factors = viz_data

                    kfs, supp_kfs = all_kfs
                    num_map_kfs = len(residuals)

                    gt_poses = [self.dataset[int(t)]['T'] for t in timestamps]
                    pred_poses = [start_pose @ to_np(pose) for pose in poses]

                    pred_poses_main, pred_poses_supp = pred_poses[:num_map_kfs], pred_poses[num_map_kfs:]
                    gt_poses_main, gt_poses_supp = gt_poses[:num_map_kfs], gt_poses[num_map_kfs:]

                    for t, pose, gt_pose in zip(timestamps, pred_poses_main, gt_poses_main):
                        self.pred_pose_dict[t] = pose
                        self.gt_pose_dict[t] = gt_pose

                    rescaled_poses = self.realign_poses(pred_poses_supp, timestamps[num_map_kfs:])
                    supp_viz_ids = ['supp_{:d}'.format(i) for i in range(len(rescaled_poses['add_pred_poses']))]
                    timestamps_viz = rescaled_poses['timestamps'] #+ supp_viz_ids

                    align_info = rescaled_poses['align_info']

                    self.set_current_residual(kfs, residuals, poses, 
                                              timestamps, factors, 
                                              supp_kfs, align_info)

                    def update_pose_render_helper():
                        self.update_pose_renders(rescaled_poses['gt_poses'], 
                                                 rescaled_poses['pred_poses'], 
                                                 rescaled_poses['timestamps'])
                        self.update_supp_pose_renders(rescaled_poses['add_pred_poses'], 
                                                      supp_viz_ids)
                        return
                    
                    gui.Application.instance.post_to_main_thread(
                        self.window, update_pose_render_helper)

                    if not pose_set:
                        pose_set = True
                        gui.Application.instance.post_to_main_thread(
                            self.window, update_pose_helper)
                    

            release_data(viz_data)
            viz_data = self.viz_queue.pop_until_latest(block=False, timeout=0.01)
            
    

            self.idx += 1

    
        self.shutdown_slam_processes()
