import numpy as np

def pose_to_camera_setup(pose, pose_init, scale):

    # Assume original negative y axis is up
    up_global = pose_init[:3,2]
    # print(up_global)
    # up_global = np.array([0, 0, 1.0])

    # Camera coordinates
    center = scale*np.array([0, 0.0, 0.5]) # Point camera is looking at
    eye = scale*np.array([0, -0.0, -0.5]) # Camera location

    def rot2eul(R):
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        return np.array((alpha, beta, gamma))

    def eul2rot(theta):
        R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                      [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                      [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
        return R

    # Transform into world coordinates ()
    R = pose[:3,:3]
    t = pose[:3,3]

    zyx = rot2eul(R)
    # zyx[0] = 0.0 # Roll
    # zyx[2] = 0.0 # Pitch
    R = eul2rot(zyx)

    center = R @ center + t
    eye = R @ eye + t

    return center, eye, up_global

