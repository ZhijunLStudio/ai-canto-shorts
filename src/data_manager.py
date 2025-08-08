# src/data_manager.py (V3.4 - High-Fidelity Summary)

import json
import logging
from typing import List, Dict, Any

class VideoDataManager:
    def __init__(self):
        self.video_chunks: List[Dict[str, Any]] = []
        self.video_data_by_second: List[Dict[str, Any]] = []
        self.audio_data_by_dialogue: List[Dict[str, Any]] = []
        self.is_initialized = False
        self.logger = logging.getLogger("DataManager")
        self.skip_seconds = 0

    def initialize_from_files(self, video_json_path: str, audio_json_path: str, skip_seconds: int = 0):
        if self.is_initialized:
            self.logger.warning("DataManager is already initialized. Skipping re-initialization.")
            return

        self.logger.info("Initializing DataManager...")
        self.skip_seconds = skip_seconds
        
        try:
            with open(video_json_path, 'r', encoding='utf-8') as f:
                raw_video_chunks = json.load(f)
                for chunk in raw_video_chunks:
                    if chunk.get('start_time', 0) >= self.skip_seconds:
                        if 'breakdown' in chunk and isinstance(chunk['breakdown'], list):
                            chunk['breakdown'] = [item for item in chunk['breakdown'] if item.get('second', 0) >= self.skip_seconds]
                            self.video_data_by_second.extend(chunk['breakdown'])
                        self.video_chunks.append(chunk)

            with open(audio_json_path, 'r', encoding='utf-8') as f:
                raw_audio_data = json.load(f)
                self.audio_data_by_dialogue = [item for item in raw_audio_data if item.get('start_time', 0) >= self.skip_seconds]
            
            self.is_initialized = True
            log_msg = f"✅ DataManager initialized"
            if self.skip_seconds > 0:
                log_msg += f", skipping first {self.skip_seconds} seconds of content."
            self.logger.info(log_msg)

        except Exception as e:
            self.logger.critical(f"❌ An error occurred during DataManager initialization: {e}", exc_info=True)
            raise

    def get_total_duration_seconds(self) -> int:
        if not self.is_initialized: return 0
        last_video_sec = max([item.get('second', 0) for item in self.video_data_by_second], default=0)
        last_audio_sec = max([item.get('end_time', 0) for item in self.audio_data_by_dialogue], default=0)
        return int(max(last_video_sec, last_audio_sec))

    def get_content_by_time(self, start_sec: int, end_sec: int) -> str:
        if not self.is_initialized: return "Error: DataManager not initialized."
        visual_content_list: List[str] = [
            f"[{s['second']}s]: {s['description']} (说话人: {s.get('speaker', '未知')})"
            for s in self.video_data_by_second 
            if start_sec <= s.get('second', -1) < end_sec
        ]
        audio_content_list: List[str] = [
            f"({s['start_time']}s-{s['end_time']}s) 对白: {s['origin_text']}" 
            for s in self.audio_data_by_dialogue 
            if max(start_sec, s['start_time']) < min(end_sec, s['end_time'])
        ]
        visual_str = "\n".join(visual_content_list) if visual_content_list else "此时间段无具体视觉事件描述。"
        audio_str = "\n".join(audio_content_list) if audio_content_list else "此时间段无对白。"
        return f"--- 时间段 [{start_sec}s - {end_sec}s] ---\n【视觉描述】:\n{visual_str}\n\n【对白内容】:\n{audio_str}\n--- 数据结束 ---"
        
    def get_full_summary(self) -> str:
        if not self.is_initialized:
            return "错误：数据管理器未初始化。"

        visual_summaries = [f"- (时间段 {chunk.get('start_time')}s-{chunk.get('end_time')}s): {chunk.get('summary', '无摘要')}" for chunk in self.video_chunks]
        visual_summary_str = "\n".join(visual_summaries) if visual_summaries else "无视觉摘要信息。"

        detailed_breakdown_str = "\n".join([f"[{s['second']}s]: {s['description']}" for s in self.video_data_by_second])
        audio_dialogues_str = "\n".join([f"({s['start_time']}s-{s['end_time']}s): {s['origin_text']}" for s in self.audio_data_by_dialogue])

        full_text = (
            "以下是整个视频的完整内容分析，分为【整体视觉摘要】、【逐秒视觉描述】和【全部对话】三部分。\n\n"
            "--- 【整体视觉摘要】 ---\n"
            f"{visual_summary_str}\n\n"
            "--- 【逐秒视觉描述】 ---\n"
            f"{detailed_breakdown_str}\n\n"
            "--- 【按时间顺序排列的全部对话】 ---\n"
            f"{audio_dialogues_str}\n"
        )
        return full_text

data_manager = VideoDataManager()