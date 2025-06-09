"""
trajectory_matching.py

This module implements multi-modal matching and trajectory-to-language description utilities.
It includes functions for matching trajectories using CLIP, LLM, and token-based approaches,
as well as generating trajectory-to-language descriptions.
"""

import torch
import numpy as np
from model.llm.llm_utils import ask_llm
from model.matching.clip_utils import clip_encode_text
from model.matching.similarity import calculate_lidar_clip_similarity_score
from model.tracker.trajectory_utils import generate_movement_description
from model.common.utils import real_world_to_image_domain_and_2d

# --- Utility Functions ---
def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    deno = float(box1_area + box2_area - inter_area)
    if deno == 0:
        return 0.0
    return inter_area / deno

def parse_bbox(bbox_str):
    """Parse a bbox string of the form 'x1_y1_x2_y2' into a tuple of floats."""
    return tuple(map(float, bbox_str.split('_')))

# --- Main Matching Logic ---
def multimodal_trajectory_matching(context):
    """
    Given a ProcessTrajectoryContext, perform multi-modal matching and trajectory-to-language description.
    Returns: similarity, answer, rounded_detection_2d
    """
    # Get last trajectory state
    last_five_elements = list(context.track.trajectory.items())[-6:]
    for _, data in last_five_elements:
        trajectory_data = data.updated_state if data.updated_state is not None else data.predicted_state
    state_vector = torch.tensor([
        trajectory_data[0, 0], trajectory_data[1, 0], trajectory_data[2, 0],
        trajectory_data[9, 0], trajectory_data[10, 0], trajectory_data[11, 0], trajectory_data[12, 0],
    ])
    _, detections_box_2d, ego_translation_coord = real_world_to_image_domain_and_2d(
        state_vector.unsqueeze(0), context.data.v2c, context.data.p2, context.data.current_pose
    )
    detection_2d_det = detections_box_2d[0]
    rounded_detection_2d = [round(value.item(), 0) for value in detection_2d_det]
    key = "_".join(map(str, rounded_detection_2d))
    # LLM output selection
    if not context.tracker.config.query_tokenization:
        try:
            llm_output_det = context.llm_output[key]
        except:
            best_iou = 0
            best_match = None
            bbox1 = parse_bbox(key)
            try:
                for bbox2_str, desc2 in context.llm_output_2.items():
                    bbox2 = parse_bbox(bbox2_str)
                    iou = compute_iou(bbox1, bbox2)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = bbox2_str
                if best_match and best_iou >= 0.6:
                    llm_output_det = context.llm_output_2[best_match]
                else:
                    llm_output_det = None
            except:
                llm_output_det = None
    else:
        llm_output_det = None
    # Matching and description
    if llm_output_det is None or context.tracker.config.use_llm:
        car_trajectory = {}
        x_list, y_list, z_list, heading_list = [], [], [], []
        for _, data in last_five_elements:
            trajectory_data = data.updated_state if data.updated_state is not None else data.predicted_state
            x = round(trajectory_data[1, 0], 3)
            z = round(trajectory_data[0, 0], 3)
            y = round(trajectory_data[2, 0], 3)
            heading = round(trajectory_data[12, 0], 3)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            heading_list.append(((heading - np.pi / 2) % (2 * np.pi) - np.pi))
        start_x, start_y, start_z = x_list[0], y_list[0], z_list[0]
        end_x, end_y, end_z = x_list[-1], y_list[-1], z_list[-1]
        euclidean_distance_difference = round(
            ((end_x - start_x) ** 2 + (end_y - start_y) ** 2 + (end_z - start_z) ** 2) ** 0.5, 3
        )
        heading_variations = [
            round(abs(heading_list[i] - heading_list[i - 1]), 3)
            for i in range(1, len(heading_list))
        ]
        mean_heading = round(sum(heading_list) / len(heading_list), 3)
        mean_heading_variation = round(sum(heading_variations) / len(heading_variations), 3)
        x_variation = round(max(x_list) - min(x_list), 3)
        y_variation = round(max(y_list) - min(y_list), 3)
        z_variation = round(max(z_list) - min(z_list), 3)
        car_trajectory = {
            "x=": ego_translation_coord[0, 1].item(),
            "y=": ego_translation_coord[0, 2].item(),
            "z=": ego_translation_coord[0, 0].item(),
            "average_direction_angle=": mean_heading,
            "euclidean_distance_difference=": euclidean_distance_difference,
            "mean_heading_variation=": mean_heading_variation,
            "x_variation=": x_variation,
            "y_variation=": y_variation,
            "z_variation=": z_variation
        }
        movement_description = ask_llm(car_trajectory, 
                                       context.data.image, 
                                       context.tracker.config, 
                                       detection_2d=detection_2d_det, 
                                       vlm_model=context.vlm_model,
                                       vlm_processor=context.vlm_processor)
    else:
        movement_description = llm_output_det
    similarity = calculate_lidar_clip_similarity_score(
        context.tracker.expression,
        movement_description,
        context.tracker.query_embedding,
        context.models.clip_model
    )
    return similarity, movement_description, rounded_detection_2d