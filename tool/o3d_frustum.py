import open3d as o3d
import numpy as np

class Frustum():

    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0., -2.5, -50.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]
 
        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]


        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

 

def create_frustum(pose, frusutum_color=[0, 1, 0]):
    size = 0.02
    points = np.array([[0., 0., 0], [1.0, -0.5, 2], [-1.0, -0.5, 2],
                       [1.0, 0.5, 2], [-1.0, 0.5, 2]]) * size

 
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [1, 3],
             [2, 4],
             [3, 4]]

    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)

    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)

    return frustum

 