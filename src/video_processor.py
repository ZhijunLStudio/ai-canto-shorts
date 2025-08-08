# src/video_processor.py (V7.0 Final - Based on User's V4.0)

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from tqdm import tqdm

class SmartVideoProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing SmartVideoProcessor (V7.0 Final)...")
        self.model = YOLO(self.config.MODEL_PATH)
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    def _analyze_video(self, video_path, segments_info):
        pose_data_file = os.path.join(self.config.RUN_BASE_DIR, 'pose_tracking_data.json')
        if os.path.exists(pose_data_file):
            self.logger.info(f"Pose tracking data file found at '{pose_data_file}', skipping analysis.")
            with open(pose_data_file, 'r') as f:
                return json.load(f)
        self.logger.info("--- Starting video analysis (Pose Tracking) ---")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Error: Could not open video file {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_pose_data = {}
        for i, segment in enumerate(tqdm(segments_info, desc="Analyzing Segments")):
            start_time, end_time = segment['start_time'], segment['end_time']
            start_frame, end_frame = int(start_time * fps), int(end_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            segment_poses = {}
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret: break
                results = self.model.track(frame, persist=True, verbose=False, classes=[0])
                frame_detections = []
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes, track_ids, kpts = results[0].boxes.xywh.cpu().numpy(), results[0].boxes.id.int().cpu().tolist(), results[0].keypoints.data.cpu().numpy()
                    for box, track_id, kpt in zip(boxes, track_ids, kpts):
                        frame_detections.append({'id': track_id, 'box': box.tolist(), 'keypoints': kpt.tolist()})
                segment_poses[str(frame_num)] = frame_detections
            all_pose_data[f"segment_{i}"] = segment_poses
        cap.release()
        with open(pose_data_file, 'w') as f:
            json.dump(all_pose_data, f, indent=2)
        self.logger.info("--- Pose Tracking analysis complete. Pose data saved. ---")
        return all_pose_data

    def _preprocess_segments(self, segments_info, all_pose_data, fps):
        self.logger.info("--- Starting segment pre-processing ---")
        filtered_segments, original_indices = [], []
        for i, segment in enumerate(segments_info):
            has_any_person = any(d for d in all_pose_data.get(f"segment_{i}", {}).values())
            self.logger.info(f"Segment {i+1}: {'Person detected' if has_any_person else 'No person detected'}. Kept.")
            filtered_segments.append(segment.copy()); original_indices.append(i)
        self.logger.info(f"--- Pre-processing complete. Kept {len(filtered_segments)}/{len(segments_info)} segments. ---")
        return filtered_segments, original_indices
        
    def _get_precise_target_x(self, person_data):
        kpts = np.array(person_data['keypoints'])
        h_kpts = self.config.HEAD_KEYPOINT_INDICES
        # Priority 1: Nose
        if kpts[h_kpts['nose'], 2] > self.config.MIN_KEYPOINT_CONFIDENCE:
            return kpts[h_kpts['nose'], 0]
        # Priority 2: Visible Eyes
        visible_eyes_x = [k[0] for i, k in enumerate(kpts) if i in [h_kpts['left_eye'], h_kpts['right_eye']] and k[2] > self.config.MIN_KEYPOINT_CONFIDENCE]
        if visible_eyes_x: return np.mean(visible_eyes_x)
        # Priority 3: Visible Ears
        visible_ears_x = [k[0] for i, k in enumerate(kpts) if i in [h_kpts['left_ear'], h_kpts['right_ear']] and k[2] > self.config.MIN_KEYPOINT_CONFIDENCE]
        if visible_ears_x: return np.mean(visible_ears_x)
        # Fallback: Bounding box center
        return person_data['box'][0]

    def _calculate_camera_paths(self, video_path, segments_info, all_pose_data, original_indices):
        self.logger.info("--- Starting smart camera path calculation (V7.0 Final Logic) ---")
        cap = cv2.VideoCapture(video_path)
        h, w, fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        crop_width = int(h * self.config.TARGET_ASPECT_RATIO)
        self.logger.info(f"Video properties: {w}x{h} @ {fps:.2f}fps. Vertical crop width: {crop_width}px")

        all_segment_camera_moves, successfully_processed_segments = {}, []

        for i, (segment, original_idx) in enumerate(zip(segments_info, original_indices)):
            segment_key = f"segment_{original_idx}"
            self.logger.info(f"Calculating path for segment {i+1}/{len(segments_info)} ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s)")
            
            segment_poses = all_pose_data.get(segment_key, {})
            start_frame_abs, end_frame_abs = int(segment['start_time'] * fps), int(segment['end_time'] * fps)
            num_frames_in_segment = end_frame_abs - start_frame_abs
            if num_frames_in_segment <= 0: continue
            
            final_camera_positions = np.full(num_frames_in_segment, w / 2.0)
            camera_x = w / 2.0
            
            if not any(detections for detections in segment_poses.values()):
                if self.config.ADAPTIVE_KEN_BURNS_ENABLED:
                    final_camera_positions[:] = np.linspace(w/2 - crop_width/4, w/2 + crop_width/4, num_frames_in_segment)
            else:
                protagonist_id, last_protagonist_id, protagonist_tenure_frames = None, None, 0
                min_shot_duration_frames = int(self.config.MIN_SHOT_DURATION_SECONDS * fps)
                keypoint_history = defaultdict(lambda: deque(maxlen=self.config.ACTIVITY_HISTORY_SIZE))
                activity_scores = defaultdict(float)
                
                for frame_idx, frame_num in enumerate(range(start_frame_abs, end_frame_abs)):
                    raw_detections = segment_poses.get(str(frame_num), [])
                    valid_detections = [d for d in raw_detections if np.sum(np.array(d['keypoints'])[:, 2] > self.config.MIN_KEYPOINT_CONFIDENCE) >= self.config.MIN_VISIBLE_KEYPOINTS]
                    
                    if valid_detections:
                        current_track_ids = {d['id'] for d in valid_detections}
                        for det in valid_detections:
                            track_id, kpts = det['id'], np.array(det['keypoints'])[:,:2]
                            keypoint_history[track_id].append(kpts)
                            if len(keypoint_history[track_id]) > 5:
                                diff = np.linalg.norm(kpts - np.mean(keypoint_history[track_id], axis=0), axis=1)
                                weighted_diff = sum(diff[k_idx] * self.config.KEYPOINT_WEIGHTS.get(k_idx, self.config.DEFAULT_WEIGHT) for k_idx in range(len(diff)))
                                activity_scores[track_id] = activity_scores[track_id] * 0.7 + weighted_diff * 0.3
                        
                        challenger_id = max(current_track_ids, key=lambda tid: activity_scores.get(tid, 0))
                        if (protagonist_id is None or protagonist_id not in current_track_ids) or \
                           (protagonist_tenure_frames >= min_shot_duration_frames and challenger_id != protagonist_id and \
                            activity_scores.get(challenger_id, 0) > activity_scores.get(protagonist_id, 0) * self.config.PROTAGONIST_SWITCH_THRESHOLD):
                            protagonist_id = challenger_id
                        
                        protagonist_has_switched = (protagonist_id is not None and last_protagonist_id is not None and protagonist_id != last_protagonist_id)
                        if protagonist_id == last_protagonist_id:
                            protagonist_tenure_frames += 1
                        else:
                            protagonist_tenure_frames = 0
                        
                        main_person = next((d for d in valid_detections if d['id'] == protagonist_id), None)

                        if main_person:
                            target_x = self._get_precise_target_x(main_person)
                            if protagonist_has_switched:
                                camera_x = target_x # Hard Cut
                            else:
                                camera_x = self.config.CAMERA_SMOOTHING_FACTOR * target_x + (1 - self.config.CAMERA_SMOOTHING_FACTOR) * camera_x
                    
                    final_camera_positions[frame_idx] = np.clip(camera_x, crop_width / 2.0, w - crop_width / 2.0)
                    last_protagonist_id = protagonist_id

            all_segment_camera_moves[segment_key] = final_camera_positions.tolist()
            segment_with_meta = segment.copy()
            segment_with_meta.update({'width': w, 'height': h, 'fps': fps, 'start_frame': start_frame_abs, 'end_frame': end_frame_abs})
            successfully_processed_segments.append(segment_with_meta)
            self.logger.info(f"  -> Path calculation complete.")
            
        return all_segment_camera_moves, successfully_processed_segments
        
    def _create_debug_video(self, video_path, segments_info, all_pose_data, all_segment_camera_moves, original_indices):
        self.logger.info("--- Starting debug video generation (per-clip) ---")
        cap_read = cv2.VideoCapture(video_path)
        if not cap_read.isOpened(): raise IOError(f"Error opening video for debug: {video_path}")
        
        fps, w, h = cap_read.get(cv2.CAP_PROP_FPS), int(cap_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop_width = int(h * self.config.TARGET_ASPECT_RATIO)

        for i, (segment, original_idx) in enumerate(zip(segments_info, original_indices)):
            segment_key = f"segment_{original_idx}"
            camera_moves = all_segment_camera_moves.get(segment_key)
            if not camera_moves: continue
            
            output_path = os.path.join(self.config.DEBUG_VIDEOS_DIR, f"debug_clip_{original_idx}_{segment['start_time']}s_{segment['end_time']}s.mp4")
            self.logger.info(f"Generating debug video: {output_path}")

            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            start_frame_abs, end_frame_abs = int(segment['start_time']*fps), int(segment['end_time']*fps)
            for frame_idx, frame_num in enumerate(tqdm(range(start_frame_abs, end_frame_abs), desc=f"Writing debug clip {i+1}")):
                cap_read.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap_read.read()
                if not ret: break
                
                raw_detections = all_pose_data.get(segment_key, {}).get(str(frame_num), [])
                for det in raw_detections:
                    track_id, box_xywh, kpts = det['id'], det['box'], np.array(det['keypoints'])
                    color = self.colors[track_id % len(self.colors)]
                    x_center, y_center, box_w, box_h = box_xywh
                    x1, y1, x2, y2 = int(x_center-box_w/2), int(y_center-box_h/2), int(x_center+box_w/2), int(y_center+box_h/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    for px, py, pconf in kpts:
                        if pconf > self.config.MIN_KEYPOINT_CONFIDENCE:
                            cv2.circle(frame, (int(px), int(py)), 5, color, -1)
                    for start_p, end_p in self.skeleton:
                        if start_p-1 < len(kpts) and end_p-1 < len(kpts) and kpts[start_p-1, 2] > self.config.MIN_KEYPOINT_CONFIDENCE and kpts[end_p-1, 2] > self.config.MIN_KEYPOINT_CONFIDENCE:
                            start_point = tuple(kpts[start_p-1, :2].astype(int))
                            end_point = tuple(kpts[end_p-1, :2].astype(int))
                            cv2.line(frame, start_point, end_point, color, 2)
                
                if frame_idx < len(camera_moves):
                    cam_x = camera_moves[frame_idx]
                    crop_x1, crop_x2 = int(max(0, cam_x-crop_width/2)), int(min(w, cam_x+crop_width/2))
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (crop_x1, 0), (crop_x2, h), (0, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    cv2.rectangle(frame, (crop_x1, 0), (crop_x2, h), (0, 255, 255), 4)
                
                writer.write(frame)
            writer.release()
        cap_read.release()
        self.logger.info("--- Debug video generation complete. ---")