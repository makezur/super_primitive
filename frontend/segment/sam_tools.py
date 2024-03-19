import torch
import tool.point_utils as point_utils

from segment_anything import sam_model_registry, SamPredictor

import numpy as np

def setup_sam(sam_checkpoint="./checkpoints/sam_vit_h_4b8939.pth",
               device="cuda"):

    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor

def infer_sam_masks_batch(model_sam, image, points, logits=True):
    # points is same form as keypoints, [N, 2]
    # in the range [-1, 1]
    device = points.device
    num_pts = points.shape[0]

    model_sam.set_image(image)
    
    H, W, _ = image.shape 
    
    H_sam, W_sam = model_sam.transform.get_preprocess_shape(H, W, model_sam.transform.target_length)

    points_sam_format = point_utils.denormalise_coordinates(points, (H_sam, W_sam)) 
    points_sam_format = points_sam_format.flip(-1)

    # sam ids for postive / negative ponits. here we use one postive per segment 
    dummy_ids = torch.ones((num_pts, 1), dtype=torch.int64, device=device)

    masks, iou_predictions, lowres =  model_sam.predict_torch(points_sam_format[:, None],
                                                              dummy_ids,
                                                              multimask_output=True,
                                                              return_logits=logits)
    
    return {'masks': masks, 
            'iou_pred': iou_predictions,
            'lowres': lowres}
