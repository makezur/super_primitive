import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage as nd

def fill_depth(depth, invalid_mask):
    ind = nd.distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    return depth[tuple(ind)]

def fill_single_griddata(depths_zbuff, pred_invalid_np):
    depths_zbuff[pred_invalid_np] = np.NaN
    
    x, y = np.indices(depths_zbuff.shape)
    interp = np.array(depths_zbuff)
    interp[np.isnan(interp)] = griddata(
        (x[~np.isnan(depths_zbuff)], y[~np.isnan(depths_zbuff)]), # points we know
        depths_zbuff[~np.isnan(depths_zbuff)],                    # values we know
        (x[np.isnan(depths_zbuff)], y[np.isnan(depths_zbuff)]))   # points to interpolate

    interp = fill_depth(interp, np.isnan(interp))

    return interp
