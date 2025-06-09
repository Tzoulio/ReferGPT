import re
import torch
import json
import numpy as np
import os
from PIL import Image
from model.common.config import cfg, cfg_from_yaml_file
from model.matching.clip_utils import encode_full_image, clip_encode_text
from model.llm.llm_utils import ask_llm
from model.matching.similarity import calculate_lidar_clip_similarity_score
from model.matching.trajectory_matching import multimodal_trajectory_matching
from .tracker.trajectory_utils import generate_movement_description
from .tracker.trajectory import Trajectory
from .tracker.box_op import *
from dataclasses import dataclass

@dataclass
class Models:
    clip_model: object
    clip_preprocess: object

@dataclass
class DataContext:
    image: object
    p2: object
    v2c: object
    current_pose: object

@dataclass
class ProcessTrajectoryContext:
    tracker: object  # Reference to the Tracker3D instance (for config, expression, query_embedding, etc.)
    track: object
    models: Models
    data: DataContext
    llm_output: object
    llm_output_2: object
    vlm_model: object
    vlm_processor: object

class Tracker3D:
    def __init__(self,tracking_features=False,
                    bb_as_features=False,
                    box_type='Kitti',
                    video_id=None,
                    config = None, expression = None, clip_model=None, clip_preprocess=None, t5_model=None, t5_tokenizer=None, vlm_model=None, vlm_processor=None):
        """
        initialize the the 3D tracker
        Args:
            tracking_features: bool, if tracking the features
            bb_as_features: bool, if tracking the bbs
            box_type: str, box type, available box type "OpenPCDet", "Kitti", "Waymo"
        """
        self.video_id = video_id
        self.config = config
        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type
        self.expression = expression
        self.clip_model, self.clip_preprocess = clip_model, clip_preprocess
        self.t5_model, self.t5_tokenizer = t5_model, t5_tokenizer
        self.vlm_model, self.vlm_processor = vlm_model, vlm_processor
        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

        self.query_embedding = clip_encode_text(self.expression, self.clip_model, "cuda")
        if self.config.query_tokenization:
            self.query_structure_tokenization = self.query_to_tokenized_query(self.expression, self.t5_model, self.t5_tokenizer)
        else: self.query_structure_tokenization = None
    def tracking(self,bbs_3D = None,
                 features = None,
                 scores = None,
                 pose = None,
                 timestamp = None,
                 v2c=None,
                 p2=None,
                 image=None
                 ):
        """
        tracking the objects at the given timestamp
        Args:
            bbs: array(N,7) or array(N，7*k), 3D bounding boxes or 3D tracklets
                for tracklets, the boxes should be organized to [[box_t; box_t-1; box_t-2;...],...]
            features: array(N,k), the features of boxes or tracklets

            scores: array(N,), the detection score of boxes or tracklets
            pose: array(4,4), pose matrix to global scene
            timestamp: int, current timestamp, note that the timestamp should be consecutive

        Returns:
            bbs: array(M,7), the tracked bbs
            ids: array(M,), the assigned IDs for bbs
        """
        self.current_bbs = bbs_3D
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp
        self.v2c=v2c
        self.p2 = p2
        self.image = image

        self.trajectores_prediction()

        if self.current_bbs is None:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            if len(self.current_bbs) == 0:
                return np.zeros(shape=(0,7)),np.zeros(shape=(0))

            else:
                self.current_bbs = convert_bbs_type(self.current_bbs,self.box_type)
                self.current_bbs = register_bbs(self.current_bbs,self.current_pose)
                ids = self.association()
                bbs,ids = self.trajectories_update_init(ids)

                return np.array(bbs),np.array(ids)


    def trajectores_prediction(self):
        """
        predict the possible state of each active trajectories, if the trajectory is not updated for a while,
        it will be deleted from the active trajectories set, and moved to dead trajectories set
        Returns:

        """
        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                if self.active_trajectories[key].consecutive_missed_num>=self.config.max_prediction_num:
                    dead_track_id.append(key)
                    continue
                if len(self.active_trajectories[key])-self.active_trajectories[key].consecutive_missed_num == 1 \
                    and len(self.active_trajectories[key])>= self.config.max_prediction_num_for_new_object :
                    dead_track_id.append(key)
                self.active_trajectories[key].state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra

    def compute_cost_map(self):
        """
        compute the cost map between detections and predictions
        Returns:
              cost, array(N,M), where N is the number of detections, M is the number of active trajectories
              all_ids, list(M,), the corresponding IDs of active trajectories
        """
        all_ids = []

        all_predictions = []
        all_detections = []

        for key in self.active_trajectories.keys():
            all_ids.append(key)
            state = np.array(self.active_trajectories[key].trajectory[self.current_timestamp].predicted_state)
            state = state.reshape(-1)

            pred_score = np.array([self.active_trajectories[key].trajectory[self.current_timestamp].prediction_score])
            consec_miss = np.array([self.active_trajectories[key].consecutive_missed_num])
            current_timestamp = np.array([self.current_timestamp])
            state = np.concatenate([state,pred_score, consec_miss, current_timestamp])
            all_predictions.append(state)

        for i in range(len(self.current_bbs)):
            box = self.current_bbs[i]
            noise_level = 0.0
            noise = np.random.normal(loc=0.0, scale=noise_level, size=(2,))

            noisy_box = box.copy()
            noisy_box[:2] += noise

            box = noisy_box
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]
            label=1
            new_tra = Trajectory(init_bb=box,
                                 init_features=features,
                                 init_score=score,
                                 init_timestamp=self.current_timestamp,
                                 label=label,
                                 tracking_features=self.tracking_features,
                                 bb_as_features=self.bb_as_features,
                                 config = self.config)

            state = new_tra.trajectory[self.current_timestamp].predicted_state
            state = state.reshape(-1)
            all_detections.append(state)
    
        all_detections = np.array(all_detections)
        all_predictions = np.array(all_predictions)

        det_len = len(all_detections)
        pred_len = len(all_predictions)
    
        all_detections = np.tile(all_detections,(1,pred_len,1))
        all_predictions = np.tile(all_predictions,(det_len,1,1))
        dis = (all_detections[...,0:3]-all_predictions[...,0:3])**2
        dis = np.sqrt(dis.sum(-1))

        box_cost = (dis) * all_predictions[...,-3]
        return box_cost,all_ids

    def association(self):
        """
        greedy assign the IDs for detected state based on the cost map
        Returns:
            ids, list(N,), assigned IDs for boxes, where N is the input boxes number
        """
        if len(self.active_trajectories) == 0:
            ids = []
            for i in range(len(self.current_bbs)):
                ids.append(self.label_seed)
                self.label_seed+=1
            return ids
        else:
            ids = []
            cost_map, all_ids = self.compute_cost_map()
            for i in range(len(self.current_bbs)):
                min = np.min(cost_map[i])
                arg_min = np.argmin(cost_map[i])

                if min<2.:
                    ids.append(all_ids[arg_min])
                    cost_map[:,arg_min] = 100000
                else:
                    ids.append(self.label_seed)
                    self.label_seed+=1
            return ids

    def trajectories_update_init(self, ids: list) -> tuple:
        """
        Update existing trajectories based on association results, or initialize new ones.
        Args:
            ids: list or array(N), the assigned ids for boxes
        Returns:
            valid_bbs: np.ndarray, valid bounding boxes
            valid_ids: np.ndarray, valid ids
        """
        assert len(ids) == len(self.current_bbs)
        valid_bbs = []
        valid_ids = []
        video_frame_dict = {}
        file_name = os.path.join(self.config.llm_output_data_file, f"{self.video_id}", f"{self.current_timestamp}.json")
        llm_output = self._load_json(file_name)
        file_name_2 = self._get_llm_output_file_2()
        llm_output_2 = self._load_json(file_name_2)
        for i, (label, box) in enumerate(zip(ids, self.current_bbs)):
            features = self.current_features[i] if self.current_features is not None else None
            score = self.current_scores[i]
            if label in self.active_trajectories and score > self.config.update_score:
                sim_bool, similarity, answer, rounded_detection_2d = self._update_existing_trajectory(
                    label, box, features, score, llm_output, llm_output_2
                )
                key = "_".join(map(str, rounded_detection_2d))
                video_frame_dict[key] = answer
                valid_bbs.append(box)
                valid_ids.append(label)
            elif score > self.config.init_score:
                self._init_new_trajectory(label, box, features, score)
                valid_bbs.append(box)
                valid_ids.append(label)
        self._save_llm_output(file_name, video_frame_dict)
        if not valid_bbs:
            return np.zeros((0, 7)), np.zeros((0,))
        return np.array(valid_bbs), np.array(valid_ids)

    def _load_json(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _get_llm_output_file_2(self) -> str:
        base = "/home/leandro/Documents/TrackGPT/ClipTrack/dataset/data/updated_casa_llm_output_data"
        if self.config.tracking_type == "Car":
            return os.path.join(base, f"{self.video_id}", f"{self.current_timestamp}.json")
        elif self.config.tracking_type == "Pedestrian":
            return os.path.join(base + "_pedestrian", f"{self.video_id}", f"{self.current_timestamp}.json")
        return ""

    def _update_existing_trajectory(self, label, box, features, score, llm_output, llm_output_2):
        track = self.active_trajectories[label]
        sim_bool = None
        similarity = None
        answer = None
        rounded_detection_2d = None
        if self.current_timestamp > -1:
            models = Models(
                clip_model=self.clip_model,
                clip_preprocess=self.clip_preprocess
            )
            data = DataContext(
                image=self.image,
                p2=self.p2,
                v2c=self.v2c,
                current_pose=self.current_pose
            )
            context = ProcessTrajectoryContext(
                tracker=self,
                track=track,
                models=models,
                data=data,
                llm_output=llm_output,
                llm_output_2=llm_output_2,
                vlm_model=self.vlm_model,
                vlm_processor=self.vlm_processor
            )
            similarity, answer, rounded_detection_2d = multimodal_trajectory_matching(context)
            sim_bool = similarity > self.config.similarity_threshold if similarity is not None else False
        track.state_update(
            bb=box,
            features=features,
            score=score,
            timestamp=self.current_timestamp,
            similarity=sim_bool,
            similarity_score=similarity
        )
        return sim_bool, similarity, answer, rounded_detection_2d

    def _init_new_trajectory(self, label, box, features, score):
        new_tra = Trajectory(
            init_bb=box,
            init_features=features,
            init_score=score,
            init_timestamp=self.current_timestamp,
            label=label,
            tracking_features=self.tracking_features,
            bb_as_features=self.bb_as_features,
            config=self.config
        )
        self.active_trajectories[label] = new_tra

    def _save_llm_output(self, file_name: str, video_frame_dict: dict):
        if self.config.save_llm_output:
            directory = os.path.dirname(file_name)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(file_name):
                json_data = json.dumps(video_frame_dict, indent=4)
                with open(file_name, 'w') as json_file:
                    json_file.write(json_data)

    def post_processing(self, config):
        """
        globally filter the trajectories
        Args:
            config: config

        Returns: dict(Trajectory)

        """
        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            track.filtering(config)
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            track.filtering(config)
            tra[key] = track

        return tra




