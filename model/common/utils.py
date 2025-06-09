import numpy as np
import torch
from typing import Tuple, Any
from model.tracker.box_op import bb3d_2_bb2d_torch, register_bbs

def velo_to_cam(cloud: torch.Tensor, vtc_mat: Any, device: str = 'cuda') -> torch.Tensor:
    """
    Transform velodyne (LiDAR) coordinates to camera coordinates.
    Args:
        cloud: (N, 3) or (N, 4) torch tensor of points
        vtc_mat: 4x4 transformation matrix (np.ndarray or torch.Tensor)
        device: device to use
    Returns:
        Transformed points as torch.Tensor
    """
    mat = torch.ones((cloud.shape[0], 4), dtype=torch.float32, device=device)
    mat[:, 0:3] = cloud[:, 0:3]
    if not isinstance(vtc_mat, torch.Tensor):
        vtc_mat = torch.tensor(vtc_mat, dtype=torch.float32, device=device)
    normal = vtc_mat[0:3, 0:4]
    transformed_mat = torch.matmul(normal, mat.T)
    T = transformed_mat.T.contiguous()
    return T

def real_world_to_image_domain_and_2d(
    box_template: torch.Tensor,
    v2c: Any,
    p2: Any,
    current_pose: Any,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D bounding boxes from real world to image domain and get 2D boxes.
    Args:
        box_template: (N, 7) torch tensor
        v2c: velodyne-to-camera matrix
        p2: camera projection matrix
        current_pose: current pose matrix
        device: device to use
    Returns:
        box: transformed 3D boxes
        box2d: 2D bounding boxes
        box_translation_ego: ego translation coordinates
    """
    entered_boxe = torch.clone(box_template).to(device)
    box = register_bbs(entered_boxe, torch.inverse(torch.tensor(current_pose, device=device)))
    created_boxe = torch.clone(box)
    box_translation_ego = created_boxe[:,0:3]
    box[:, 6] = -box[:, 6] - torch.pi / 2
    box[:, 2] -= box[:, 5] / 2
    box[:,0:3] = velo_to_cam(box[:,0:3], v2c, device=device)[:,0:3]
    box2d = torch.zeros((box.shape[0], 4), device=device)
    for index, pseudo_box in enumerate(box):
        pseudo_box2d = bb3d_2_bb2d_torch(pseudo_box, torch.tensor(p2[0:3,:], device=device))
        box2d[index] = pseudo_box2d
    return box, box2d, box_translation_ego