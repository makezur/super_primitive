import lietorch
import torch
from lie import lie_algebra

import copy 
def lietorch_detach(pose):
    if isinstance(pose, lietorch.LieGroupParameter):
        pose = pose.retr()
    return copy.deepcopy(pose)

def lietorch_new_param(pose):
    pose = lietorch_detach(pose)
    pose = lietorch.LieGroupParameter(pose)
    return pose

def print_pose(pose):
    pose = lietorch_detach(pose)
    print(pose.matrix())
    return 

def zero_out_lietorch_tensor(tensor):
    with torch.no_grad():
        tensor.data = torch.zeros_like(tensor.data)
    return tensor

def mat_to_lie(mat_pose):
    if torch.is_tensor(mat_pose):
        tq = lie_algebra.torch_pose_to_tq(mat_pose[None])
        current_T = lietorch.SE3.InitFromVec(tq)
    else:
        tq = lie_algebra.pose_to_tq(mat_pose[None])
        current_T = lietorch.SE3.InitFromVec(torch.from_numpy(tq).to('cuda:0').float())
    return current_T
