# src/pre_analyzer.py (V4.4 - Chinese Prompt)

import os
import json
import cv2
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
from openai import OpenAI
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

import config

class VideoAndAudioAnalyzer:
    def __init__(self):
        self.logger = tqdm.write
        self.vl_client = OpenAI(
            api_key=config.VIDEO_LLM_API_KEY, 
            base_url=config.VIDEO_LLM_BASE_URL
        )
        self.audio_chunks_by_time = None

    def _get_video_duration(self, video_path):
        try:
            with VideoFileClip(video_path) as clip:
                return clip.duration
        except Exception as e:
            self.logger(f"  - [错误] 使用 moviepy 获取视频时长失败: {e}. 尝试使用 OpenCV...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration

    def _analyze_frames_with_retry(self, frames_base64, start_s, end_s):
        relevant_audio_text = "无"
        if self.audio_chunks_by_time:
            audio_lines = [chunk['origin_text'] for chunk in self.audio_chunks_by_time if max(start_s, chunk['start_time']) < min(end_s, chunk['end_time'])]
            if audio_lines: relevant_audio_text = " ".join(audio_lines)
        
        prompt_text = f"""
你是一个精通唇语解读和场景上下文分析的、一丝不苟的逐帧视频分析AI。你的所有输出都必须是中文。
我将为你提供一个视频片段（从 {int(round(start_s))} 秒到 {int(round(end_s))} 秒）中的 {len(frames_base64)} 张连续的图像帧。
同时，我也会提供在此期间的对话内容。
你的任务是 **只返回一个原始的JSON对象**，其结构如下 (字段名保持英文，但所有字段值都必须是中文):
{{
  "summary": "【这里填写一段详细的中文段落，描述整个场景、人物和行为】",
  "breakdown": [
    {{ "second": 1, "description": "【这里填写一句完整的中文句子，描述这一帧的具体内容】", "speaker": "【见下方说明】" }}
  ]
}}
**关于 'breakdown' 的关键指令：**
1.  **一一对应：** 为我提供的 **每一帧图像** 创建一个条目。JSON中`breakdown`数组的长度必须等于 {len(frames_base64)}。
2.  **相对秒数：** `second` 字段是一个从 1 到 {len(frames_base64)} 的相对计数器，代表这是第几帧。
3.  **说话人归属 (`speaker`)：** 对于 `speaker` 字段，请分析画面和对话内容。
    - 如果画面中有人在说话，请用他们最显著的视觉特征来描述（例如：“穿西装的男士”，“戴红帽的女士”）。
    - 如果是画外音，请使用“画外音”。
    - 如果没有语音或无法判断，请使用“无”。

**此片段的上下文信息：**
- **视频帧：** 已作为图片提供。
- **此时间段内的对话：** "{relevant_audio_text}"
---
现在，请严格按照上述要求，分析这 {len(frames_base64)} 帧图像，并生成完整的、符合规范的JSON响应。
"""
        message_content = [{"type": "text", "text": prompt_text}]
        for frame_b64 in frames_base64:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
        
        max_retries = getattr(config, 'MAX_RETRIES', 3)
        retry_delay_s = getattr(config, 'RETRY_DELAY_S', 5)

        for attempt in range(max_retries):
            try:
                chat_completion = self.vl_client.chat.completions.create(
                    model=config.VIDEO_LLM_MODEL_NAME,
                    messages=[{"role": "user", "content": message_content}],
                    stream=False, max_completion_tokens=8000, temperature=0.1,
                )
                response_text = chat_completion.choices[0].message.content
                json_start_index = response_text.find('{')
                if json_start_index != -1:
                    json_end_index = response_text.rfind('}')
                    if json_end_index > json_start_index:
                        json_string = response_text[json_start_index : json_end_index + 1]
                        response_json = json.loads(json_string)
                        return {"data": response_json}
                raise json.JSONDecodeError("Response does not contain a valid JSON object.", response_text, 0)
            except json.JSONDecodeError as e:
                self.logger(f"  - [解析错误] 重试... 响应: '{e.doc[:150]}...'")
                if attempt >= max_retries - 1: return {"error": f"JSON解析失败: {e.doc[:200]}"}
                time.sleep(retry_delay_s)
            except Exception as e:
                error_type = type(e).__name__
                self.logger(f"  - [API错误: {error_type}] 重试...")
                if attempt >= max_retries - 1: return {"error": f"API请求失败: {error_type}"}
                time.sleep(retry_delay_s)
        return {"error": "未知错误"}

    def _analyze_video_chunk(self, task_info):
        video_path, start_s, end_s = task_info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": f"无法打开视频文件 {video_path}"}
        
        frames_b64 = []
        for i in range(int(end_s - start_s)):
            cap.set(cv2.CAP_PROP_POS_MSEC, (start_s + i) * 1000)
            ret, frame = cap.read()
            if not ret: continue
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
        
        if not frames_b64: return {"start_time": start_s, "end_time": end_s, "error": "未能提取任何帧。"}
        
        result = self._analyze_frames_with_retry(frames_b64, start_s, end_s)
        if "error" in result: return {"start_time": start_s, "end_time": end_s, **result}
        
        analysis_data = result["data"]
        if 'breakdown' in analysis_data and isinstance(analysis_data['breakdown'], list):
            for item in analysis_data['breakdown']:
                if 'second' in item and isinstance(item['second'], (int, float)):
                    item['second'] = int(round(start_s)) + item['second'] - 1
        analysis_data["start_time"] = round(start_s, 2)
        analysis_data["end_time"] = round(end_s, 2)
        return analysis_data

    def analyze_video_content(self, video_path):
        start_offset = getattr(config, 'SKIP_FIRST_N_SECONDS', 0)
        if self.audio_chunks_by_time is None:
            self.logger("\n[注意] 正在首先执行音频分析以提供上下文...")
            self.analyze_audio_content(video_path)

        self.logger(f"\n[视频分析] 开始使用 {config.VIDEO_LLM_MODEL_NAME} 分析视频内容... (将跳过前 {start_offset} 秒)")
        duration_s = self._get_video_duration(video_path)
        if duration_s <= start_offset: return []
            
        tasks = []
        for s in range(start_offset, int(duration_s), config.VIDEO_ANALYSIS_CHUNK_S):
            tasks.append((video_path, float(s), float(min(s + config.VIDEO_ANALYSIS_CHUNK_S, duration_s))))
        
        all_results = []
        with ThreadPoolExecutor(max_workers=config.VIDEO_ANALYSIS_CONCURRENCY) as executor:
            future_to_task = {executor.submit(self._analyze_video_chunk, task): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="分析视频内容"):
                result_item = future.result()
                if "error" in result_item:
                    self.logger(f"  - [失败] 视频块 {result_item.get('start_time', '?')}s: {result_item['error']}")
                    continue
                all_results.append(result_item)
        self.logger(f"[视频分析] 完成，共获得 {len(all_results)} 个有效分析块。")
        return sorted(all_results, key=lambda x: x['start_time'])

    def _get_asr_access_token(self):
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={config.BAIDU_TTS_API_KEY}&client_secret={config.BAIDU_TTS_SECRET_KEY}"
        try:
            response = requests.post(url); response.raise_for_status(); return response.json()['access_token']
        except Exception as e:
            self.logger(f"错误: 获取ASR access_token失败 - {e}"); return None

    def _recognize_audio_chunk(self, audio_chunk_data, token):
        request_url = f"http://vop.baidu.com/server_api?dev_pid=1637&cuid=python_video_processor&token={token}"
        headers = {'Content-Type': 'audio/pcm;rate=16000'}
        for _ in range(config.MAX_RETRIES):
            try:
                response = requests.post(request_url, data=audio_chunk_data, headers=headers, timeout=20)
                response.raise_for_status()
                result = response.json()
                if result.get("err_no") == 0: return {"text": "".join(result.get("result", [])), "error": None}
                else: return {"text": None, "error": f"API Error {result.get('err_no')}: {result.get('err_msg')}"}
            except requests.exceptions.RequestException: time.sleep(config.RETRY_DELAY_S)
        return {"text": None, "error": "Max retries exceeded"}

    def analyze_audio_content(self, video_path):
        start_offset_s = getattr(config, 'SKIP_FIRST_N_SECONDS', 0)
        self.logger(f"\n[语音分析] 开始使用 Baidu ASR 分析音频... (将跳过前 {start_offset_s} 秒)")
        access_token = self._get_asr_access_token()
        if not access_token: return []
        
        try:
            temp_dir = config.RUN_BASE_DIR or "."
            temp_audio_path = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.wav")
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', fps=16000, logger=None)
            full_audio = AudioSegment.from_wav(temp_audio_path).set_channels(1)
            os.remove(temp_audio_path)
        except Exception as e:
            self.logger(f"  - [错误] 提取音频失败: {e}"); return []

        tasks = []
        start_offset_ms = start_offset_s * 1000
        step_ms = config.AUDIO_ANALYSIS_CHUNK_MS - config.AUDIO_ANALYSIS_OVERLAP_MS
        for i in range(start_offset_ms, len(full_audio), step_ms):
            chunk = full_audio[i : i + config.AUDIO_ANALYSIS_CHUNK_MS]
            tasks.append({"start_time": i / 1000.0, "end_time": (i + config.AUDIO_ANALYSIS_CHUNK_MS) / 1000.0, "pcm_data": chunk.raw_data})
        
        results = []
        with ThreadPoolExecutor(max_workers=config.AUDIO_ANALYSIS_CONCURRENCY) as executor:
            future_to_task = {executor.submit(self._recognize_audio_chunk, task["pcm_data"], access_token): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="识别语音内容"):
                result_dict = future.result()
                if result_dict and result_dict.get("text"):
                     task = future_to_task[future]
                     results.append({"start_time": round(task["start_time"], 2), "end_time": round(task["end_time"], 2), "origin_text": result_dict["text"]})
        
        self.logger(f"[语音分析] 完成，共获得 {len(results)} 个有效语音片段。")
        self.audio_chunks_by_time = sorted(results, key=lambda x: x['start_time'])
        return self.audio_chunks_by_time