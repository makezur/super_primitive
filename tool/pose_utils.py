import numpy
import numpy as np
import copy
from tool.etc import to_np


def get_sorted_by_timestamp(pose_dict, return_ids=False):
    # sort values by keys converted to int
    sorted_by_frame_ids = sorted(pose_dict.items(), key=lambda x: int(x[0]))
    sorted_poses = [pose for _, pose in sorted_by_frame_ids]
    sorted_frame_ids = [ids for ids, pose in sorted_by_frame_ids]
    if return_ids:
        return sorted_poses, sorted_frame_ids
    return sorted_poses

def transfer_scale(gt_poses, est_poses, anchor_rotation=False):
    assert(len(est_poses) == len(gt_poses))
    assert(len(est_poses) > 0)
    
    # to np
    gt_poses = [copy.deepcopy(to_np(pose)) for pose in gt_poses]
    est_poses = [copy.deepcopy(to_np(pose)) for pose in est_poses]

    R_start_gt = copy.copy(gt_poses[0][:3,:3])
    R_start_pred = copy.copy(est_poses[0][:3,:3])

    # assert(np.allclose(R_start_gt, R_start_pred)

    # collect translations 
    gt_translations = [pose[:3,3] for pose in gt_poses]
    est_translations = [pose[:3,3] for pose in est_poses]

    # cat them into 
    gt_translations = np.stack(gt_translations, axis=1)
    est_translations = np.stack(est_translations, axis=1) 
    pose_align = align(est_translations, gt_translations)

    
    for pose_id, pose in enumerate(est_poses):
        pose[:3, 3] = pose_align['model_aligned_scaled'][:, pose_id]

        # if anchor_rotation:
        pose[:3, :3] = R_start_gt @  R_start_pred.T @ pose[:3, :3]
    
    if anchor_rotation:
        pose_align['rot_reanchor'] = R_start_gt @  R_start_pred.T

    return est_poses, pose_align

def apply_scale(T, pose_align):
    T = copy.deepcopy(to_np(T))

    s, rot, trans = pose_align['s'], pose_align['rot'], pose_align['trans_scaled']
    
    rot_reanchor = np.eye(3)
    if 'rot_reanchor' in pose_align:
        rot_reanchor = pose_align['rot_reanchor']

    T_trans = T[:3,3]
    
    T_trans = (s * (rot @ T_trans)).flatten()
    T_trans += trans.flatten()

    T[:3, 3] = T_trans

    T[:3, :3] = rot_reanchor @ T[:3, :3]

    return T


def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """


    # numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1, keepdims=True)
    data_zerocentered = data - data.mean(1, keepdims=True)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = numpy.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    
    
    transGT = data.mean(1, keepdims=True) - s*rot * model.mean(1, keepdims=True)
    trans = data.mean(1, keepdims=True) - rot * model.mean(1, keepdims=True)
    
    model_alignedGT = s*rot * model + transGT
    model_aligned = rot * model + trans

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = numpy.sqrt(numpy.sum(numpy.multiply(alignment_errorGT,alignment_errorGT),0)).A[0]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
    
    align_result = {'rot': rot, 
                    'trans_scaled': transGT,
                    'trans_scaled_error': trans_errorGT,
                    'trans': trans,
                    'trans_error': trans_error,
                    's': s,
                    'model_aligned_scaled': model_alignedGT,
                    'model_aligned': model_aligned}
    
    for key, val in align_result.items():
        if isinstance(val, numpy.matrix):
            align_result[key] = numpy.array(val)
    return align_result

