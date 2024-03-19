import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core.dense_optim as dense_optim
from tool.point_utils import to_np, img_to_np
import torch 
import cv2 

import matplotlib.colors as mcolors
import PIL 

def scatter_keypoints(image, keypoints, selected_id=0):
    # image: H x W x 3
    # keypoints: N x 2
    # selected_id: int
    # returns: H x W x 3
    image_c = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(keypoints.shape[0]):
        x, y = keypoints[i]

        if i == selected_id:
            cv2.circle(image_c, (int(x), int(y)), 4, (0, 255, 0), -1)
        else:
            cv2.circle(image_c, (int(x), int(y)), 4, (0, 0, 255), -1)

    return np.asarray(cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB))

def depth_image_lift(depth, K, image=None, mask=None, T=np.eye(4)):
    # convert K to open3d intrinsic
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)
    # convert depth to open3d image
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    # convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic, 
                                                          depth_scale=1.0, depth_trunc=100.0, stride=1)

    if image is not None:
        depth_valid = depth > 0
        pcd.colors = o3d.utility.Vector3dVector(image[depth_valid].reshape(-1, 3).astype(np.float64) / 255.0)

    if mask is not None:
        mask = mask.reshape(-1)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask, :])
        pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[mask, :])
    pcd.transform(T)

    return pcd 

def visualise_residual_batch_v2(src_keyframe, 
                        trg_image,
                        residuals,
                        residual_id=0,
                        segment_id=200, 
                        silent=False):
    if silent:
        plt.ioff()
    # disable axis visualisation 

    K = src_keyframe['K_img']
    segment_mask = residuals['segm_ids'] == segment_id
    # segment_mask = segment_mask[residual_id]

    r = residuals['residual_raw'][residual_id]

    num_pts = segment_mask.sum()

    pts_to_show_trg = dense_optim.project_points(residuals['src_in_trg_pts'][residual_id][segment_mask], 
                                                 K)
    patch_residual = torch.abs(r[:, segment_mask])
    patch_residual = torch.mean(patch_residual, dim=[0])

    each = 1
    if num_pts > 5000:
        each = 2
        if num_pts > 10000:
            each = 4

    fig = plt.figure()
    # set very tight margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    

    plt.imshow(img_to_np(trg_image))
    plt.scatter(to_np(pts_to_show_trg[:, 0][::each]),
                to_np(pts_to_show_trg[:, 1][::each]), s=1, 
                c=to_np(patch_residual)[::each], cmap='Reds')
    plt.scatter(to_np(residuals['src_in_trg_keypoints'][residual_id][segment_id, 0]),
                to_np(residuals['src_in_trg_keypoints'][residual_id][segment_id, 1]), s=4, c='b')
    fig.canvas.draw()
    img1 = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    if not silent:
        plt.show()
    plt.close()
    del fig
    return img1

def visualise_residual(src_keyframe, 
                        trg_keyframe,
                        residuals,
                        segment_id=200, 
                        silent=False):
    if silent:
        plt.ioff()
    # disable axis visualisation 

    K = src_keyframe['K_img']
    segment_mask = residuals['segm_ids'] == segment_id

    r = residuals['residual_raw']

    num_pts = segment_mask.sum()
    

    if residuals['src_in_trg_pts'].shape[-1] == 2:
        pts_to_show_trg  = residuals['src_in_trg_pts'][segment_mask]
    else:
        pts_to_show_trg = dense_optim.project_points(residuals['src_in_trg_pts'][segment_mask], 
                                                     K)
    patch_residual = torch.abs(r[:, :, segment_mask])
    # [1, 3, num_pts]
    # [1, 1, num_ptts]
    patch_residual = torch.mean(patch_residual, dim=[0, 1])

    each = 1
    if num_pts > 5000:
        each = 2
        if num_pts > 10000:
            each = 4

    fig = plt.figure()
    # set very tight margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    
    plt.imshow(img_to_np(trg_keyframe['image'][:3]))
    def to_colourmap(res):
        cmap = plt.cm.get_cmap('Reds')

        global_min = 0
        global_max = 1
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        res = norm(res)
        test = cmap(res)
        return test 

    plt.scatter(to_np(pts_to_show_trg[:, 0][::each]),
                to_np(pts_to_show_trg[:, 1][::each]), s=1, 
                c=to_colourmap(to_np(patch_residual))[::each])
    plt.scatter(to_np(residuals['src_in_trg_keypoints'][segment_id, 0]),
                to_np(residuals['src_in_trg_keypoints'][segment_id, 1]), s=4, c='b')
    fig.canvas.draw()
    img1 = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    if not silent:
        plt.show()
    plt.close()
    del fig
    return img1

def project_points_np(points_3d, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    N_pts = points_3d.shape[0]

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    u = x * fx / z + cx
    v = y * fy / z + cy

    return np.stack([u, v], axis=1)

def render_pcd(points_3d, points_colour, K, H, W):
    projected = project_points_np(points_3d, K)
    # now clip out of bounds
    projected_mask = (projected[:,0] >= 0) & (projected[:,0] < W) & (projected[:,1] >= 0) & (projected[:,1] < H)
    projected = projected[projected_mask, :]
    projected_colour = points_colour[projected_mask, :]
    image = np.zeros((H, W, 3))

    image[projected[:,1].astype(np.int64), projected[:,0].astype(np.int64)] = projected_colour
    image = (image * 255).astype(np.uint8) 
    return image