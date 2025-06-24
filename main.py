import re
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import subprocess

from model.refergpt import Tracker3D
from dataset.kitti_dataset import KittiTrackingDataset
from dataset.kitti_data_base import velo_to_cam
import time
from tqdm import tqdm
import os
from model.common.config import cfg, cfg_from_yaml_file
from model.tracker.box_op import *
import numpy as np
import argparse
import json
from evaluation_HOTA.scripts.run_kitti import eval_kitti
from model.matching.clip_utils import load_clip_model
from model.matching.similarity import calculate_lidar_clip_similarity_score
from model.llm.llm_utils import ask_llm
import clip
from sentence_transformers import SentenceTransformer
import cv2
from model.filtering.filtering_utils.filtering import dynamic_hierarchical_grouping, remove_outliers_and_scale, scale_back

from model.refergpt import Tracker3D
CLIP_VERSION = "ViT-L/14"

import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration, AutoModelForImageTextToText
from accelerate import dispatch_model, infer_auto_device_map

def launch_trackeval():
    command = [
        "python", "TrackEval/scripts/run_mot_challenge.py",
        "--ROOT_DIR", ".",
        "--METRICS", "HOTA",
        "--SEQMAP_FILE", "dataset/data_path/seqmap.txt",
        "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", "dataset/data/KITTI/training/image_02",
        "--TRACKERS_FOLDER", "evaluation/results/sha_key/data/refer-kitti",
        "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "--TRACKERS_TO_EVAL", "evaluation/results/sha_key/data/refer-kitti",
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", "2",
        "--PLOT_CURVES", "False"

    ]
    subprocess.run(command, check=True)
def track_one_seq(seq_id,config, expression, clip_model, clip_preprocess, t5_model, t5_tokenizer, vlm_model, vlm_processor):

    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        config: config
    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    """
    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)

    tracker = Tracker3D(box_type="Kitti", tracking_features=False, config = config, expression = expression, video_id=seq_id, clip_model=clip_model, clip_preprocess=clip_preprocess, t5_model=t5_model, t5_tokenizer=t5_tokenizer, vlm_model=vlm_model, vlm_processor=vlm_processor)
    dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_path,type=[tracking_type], load_image=True)

    all_time = 0
    frame_num = 0
    num_objects = 0
    for i in range(len(dataset)):
        P2, V2C, points, image, objects, det_scores, pose = dataset[i]

        mask = det_scores>config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]
        num_objects += len(objects)
        start = time.time()

        tracker.tracking(objects[:,:7],
                            features=None,
                            scores=det_scores,
                            pose=pose,
                            timestamp=i, v2c=V2C, p2=P2, image=image)
        end = time.time()
        all_time+=end-start
        frame_num+=1

    return dataset, tracker, all_time, frame_num

def save_one_seq(dataset,
                 seq_id,
                 tracker,
                 config,
                 expression):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
    """

    save_path = config.save_path
    tracking_type = config.tracking_type
    s =time.time()
    tracks = tracker.post_processing(config)
    proc_time = s-time.time()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = os.path.join(save_path,str(seq_id).zfill(4)+'.txt')

    frame_first_dict = {}
    for ob_id in tracks.keys():
        track = tracks[ob_id]

        for frame_id in track.trajectory.keys():

            ob = track.trajectory[frame_id]
            if ob.similarity is False or ob.similarity is None:
                continue
            if ob.updated_state is None:
                continue
            if ob.score<config.post_score:
                continue

            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T),ob.score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T),ob.score)}

    with open(save_name,'w+') as f:
        for i in range(len(dataset)):
            P2, V2C, points, image, _, _, pose = dataset[i]
            new_pose = np.asmatrix(pose).I
            if i in frame_first_dict.keys():
                objects = frame_first_dict[i]

                for ob_id in objects.keys():
                    updated_state,score = objects[ob_id]

                    box_template = np.zeros(shape=(1,7))
                    box_template[0,0:3]=updated_state[0,0:3]
                    box_template[0,3:7]=updated_state[0,9:13]

                    box = register_bbs(box_template,new_pose)

                    box[:, 6] = -box[:, 6] - np.pi / 2
                    box[:, 2] -= box[:, 5] / 2
                    box[:,0:3] = velo_to_cam(box[:,0:3],V2C)[:,0:3]

                    box = box[0]

                    box2d = bb3d_2_bb2d(box,P2)

                    print('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (i,ob_id,tracking_type,box2d[0][0],box2d[0][1],box2d[0][2],
                             box2d[0][3],box[5],box[4],box[3],box[0],box[1],box[2],box[6],score),file = f)
    return proc_time


def save_one_seq_with_expression(dataset,
                 seq_id,
                 tracker,
                 config,
                 expression,
                 json_filename):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
    """
    json_filename = json_filename.rsplit('.', 1)[0]
    json_filename = re.sub(r",", "", json_filename)
    save_path = config.save_path
    tracking_type = config.tracking_type
    s =time.time()
    tracks = tracker.post_processing(config)
    proc_time = s-time.time()
    if not os.path.exists(os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename)):
        os.makedirs(os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename))

    save_name = os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename, 'predict.txt')
    save_name_score = os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename, 'sim_scores.txt')

    output_folder = os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename)
    frame_first_dict = {}

    all_sim_score = []
    for ob_id in tracks.keys():
        track = tracks[ob_id]
        for frame_id in track.trajectory.keys():
            ob = track.trajectory[frame_id]
            all_sim_score.append(ob.similarity_score)
    all_sim_score_scaled, min_val, max_val = remove_outliers_and_scale(all_sim_score)
    groups, group_with_highest_mean, group_with_lowest_mean = dynamic_hierarchical_grouping(all_sim_score_scaled, config.distance_threshold, linkage_method='ward')
    min_sim_score = min(group_with_lowest_mean)

    high_min_sim_score = min(group_with_highest_mean)
    high_min_sim_score = high_min_sim_score
    max_sim_score=  max(group_with_highest_mean)
    
    scaled_high_min_sim_score = scale_back(high_min_sim_score, min_val, max_val)
    scaled_max_sim_score = scale_back(max_sim_score, min_val, max_val)
    
    for ob_id in tracks.keys():
        track = tracks[ob_id]
        mean_score = (scaled_high_min_sim_score + scaled_max_sim_score) / 2

        count_in_range = 0
        total_count = 0

        for frame_id in track.trajectory.keys():
            ob = track.trajectory[frame_id]
            if ob.similarity_score is not None:
                total_count += 1
                if scaled_high_min_sim_score <= ob.similarity_score <= scaled_max_sim_score:
                    count_in_range += 1

        if count_in_range > total_count*config.majority_voting_ratio:
            for frame_id in track.trajectory.keys():
                ob = track.trajectory[frame_id]
                ob.similarity_score = mean_score
        else:
            for frame_id in track.trajectory.keys():
                ob = track.trajectory[frame_id]
                ob.similarity_score = 0

        for frame_id in track.trajectory.keys():
            ob = track.trajectory[frame_id]
 
            if ob.updated_state is None:
                continue
            if ob.score<config.post_score:
                continue
            if ob.similarity_score is not None:
                if scaled_high_min_sim_score > ob.similarity_score or ob.similarity_score > scaled_max_sim_score:
                    continue

            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T),ob.score, ob.similarity_score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T),ob.score, ob.similarity_score)}

    with open(save_name,'w+') as f:
        for i in range(len(dataset)):
            P2, V2C, points, image, _, _, pose = dataset[i]
            new_pose = np.asmatrix(pose).I
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
            if i in frame_first_dict.keys():
                objects = frame_first_dict[i]

                for ob_id in objects.keys():
                    updated_state,score, sim_score = objects[ob_id]

                    box_template = np.zeros(shape=(1,7))
                    box_template[0,0:3]=updated_state[0,0:3]
                    box_template[0,3:7]=updated_state[0,9:13]

                    box = register_bbs(box_template,new_pose)

                    box[:, 6] = -box[:, 6] - np.pi / 2
                    box[:, 2] -= box[:, 5] / 2
                    box[:,0:3] = velo_to_cam(box[:,0:3],V2C)[:,0:3]

                    box = box[0]

                    box2d = bb3d_2_bb2d(box,P2)
                    x_min = box2d[0][0]
                    y_min = box2d[0][1]
                    x_max = x_min + box2d[0][2]  # x_min + width
                    y_max = y_min + box2d[0][3]  # y_min + height

                    x_min, y_min = int(x_min), int(y_min)
                    x_max, y_max = int(x_max), int(y_max)

                    if i > 0:
                        print('%d,%d,%.4f,%.4f,%.4f,%.4f,1,1,1'
                            % (i,ob_id,box2d[0][0],box2d[0][1],box2d[0][2],
                                box2d[0][3]),file = f)
    return proc_time

def write_gt(config, seq_id, json_filename, json_file, gt_txt_file, im_height, im_width):
    save_path = config.save_path

    json_filename = json_filename.rsplit('.', 1)[0]
    tracking_type = config.tracking_type
    s =time.time()
    proc_time = s-time.time()
    if not os.path.exists(os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename)):
        os.makedirs(os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename))

    json_filename = re.sub(r",", "", json_filename)
    txt_path = os.path.join(save_path, config.dataset_type,str(seq_id).zfill(4), json_filename, 'gt.txt')

    save_format = '{frame},{id},{x1},{y1},{w},{h},1, 1, 1\n'

    with open(json_file) as f:
        json_info = json.load(f)

    with open(txt_path, 'w') as f:
        for k in json_info['label'].keys():
            frame_id = int(k) + 1
            if not os.path.isfile(os.path.join(gt_txt_file, '{:06d}.txt'.format(frame_id))):
                continue
            frame_gt = np.loadtxt(
                os.path.join(gt_txt_file, '{:06d}.txt'.format(frame_id))).reshape(-1, 6)
            for frame_gt_line in frame_gt:
                aa = json_info['label'][k]  # all gt from frame
                aa = [int(a) for a in aa]
                if int(frame_gt_line[1]) in aa:  # choose referent gt from all gt
                    track_id = int(frame_gt_line[1])
                    x1, y1, w, h = frame_gt_line[2:6] # KITTI -> [x1, y1, w, h]
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1 * im_width, y1=y1 * im_height,
                                                w=w * im_width, h=h * im_height)
                    f.write(line)



def tracking_val_seq(config, models):

    print("data path: ", config.dataset_path)
    print('detections path: ', config.detections_path)

    save_path = config.save_path

    os.makedirs(save_path,exist_ok=True)

    seq_list = config.tracking_seqs    

    print("tracking seqs: ", seq_list)

    all_time,frame_num = 0,0

    expression_folder_path = config.expression_path
    clip_model = models['clip_model']
    clip_preprocess = models['clip_preprocess']
    t5_model = models['t5_model']
    t5_tokenizer = models['t5_tokenizer']
    vlm_model = models['vlm_model']
    vlm_processor = models['vlm_processor']

    def should_process_json(json_file, config):
        """Filter JSON files by tracking type using keyword sets."""
        name = json_file.lower()
        type_keywords = {
            'Car': {'car', 'vehicle', 'auto'},
            'Pedestrian': {'carrying'}
        }
        if config.tracking_type == 'Car':
            if not any(kw in name for kw in type_keywords['Car']):
                return False
            if 'carrying' in name:
                return False
        elif config.tracking_type == 'Pedestrian':
            if not any(kw in name for kw in type_keywords['Pedestrian']):
                if any(kw in name for kw in type_keywords['Car']):
                    return False
        return json_file.endswith('.json')

    # Main loop for processing sequences and JSON files
    for id in tqdm(range(len(seq_list)), desc="Sequences Progress"):
        seq_id = seq_list[id]
        expression_path = os.path.join(expression_folder_path, str(seq_id).zfill(4))
        json_files = [f for f in os.listdir(expression_path) if should_process_json(f, config)]
        for json_file in tqdm(json_files, desc=f"Processing JSONs for seq_id {seq_id}", leave=False):
            file_path = os.path.join(expression_path, json_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            expression = data["sentence"]
            # Use logging for better output control
            print(f"The query is: {expression}")
            dataset, tracker, this_time, this_num = track_one_seq(seq_id, config, expression, clip_model, clip_preprocess, t5_model, t5_tokenizer, vlm_model, vlm_processor)
            proc_time = save_one_seq_with_expression(dataset, seq_id, tracker, config, expression, json_file)
            write_gt(config, seq_id, json_file, file_path, os.path.join(config.gt_path, "image_02", str(seq_id).zfill(4)), 375, 1242)
            if config.save_llm_output:
                break

            break
        break
    
    
    launch_trackeval()

def load_models_from_config(config):
    """
    Load all models and processors as specified in the config.
    Returns a dict with all models and processors.
    """
    models = {}
    device = config.get('device', 'cuda')
    # CLIP
    if hasattr(config, 'clip_version'):
        import clip
        models['clip_model'], models['clip_preprocess'] = clip.load(config.clip_version, device=device)
    else:
        import clip
        models['clip_model'], models['clip_preprocess'] = clip.load("ViT-L/14", device=device)
    # T5
    if getattr(config, 'query_tokenization', False):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        models['t5_tokenizer'] = T5Tokenizer.from_pretrained(config.tokenizer_model_path)
        models['t5_model'] = T5ForConditionalGeneration.from_pretrained(config.tokenizer_model_path)
    else:
        models['t5_tokenizer'] = None
        models['t5_model'] = None
    # VLM
    if hasattr(config, 'vlm_model') and config.vlm_model == 'llava':
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        import torch
        models['vlm_model'] = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16
        )
        models['vlm_processor'] = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    # QWEN
    elif hasattr(config, 'vlm_model') and config.vlm_model == 'qwen':
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch
        models['vlm_model'] = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        models['vlm_processor'] = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    # PHI
    elif hasattr(config, 'vlm_model') and config.vlm_model == 'phi':
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        models['vlm_model'] = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        models['vlm_processor'] = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    # GPT
    # QWEN
    elif hasattr(config, 'vlm_model') and config.vlm_model == 'qwen':
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch
        models['vlm_model'] = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        models['vlm_processor'] = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    # PHI
    elif hasattr(config, 'vlm_model') and config.vlm_model == 'phi':
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        models['vlm_model'] = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        models['vlm_processor'] = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    # GPT
    elif hasattr(config, 'vlm_model') and config.vlm_model == 'gpt4':
        models['vlm_model'] = None  # GPT-4 API is used directly
        models['vlm_processor'] = None
    else:
        models['vlm_model'] = None
        models['vlm_processor'] = None
    return models

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="config/global/cfg_refergpt.yaml",
                        help='specify the config for tracking')
    parser.add_argument('--device', type=str, default=None, help='Device to use (overrides config)')
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg_file, cfg)
    if args.device:
        config.device = args.device
    models = load_models_from_config(config)
    tracking_val_seq(config, models)

if __name__ == '__main__':
    main()

