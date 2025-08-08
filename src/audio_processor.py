# src/audio_processor.py (V3.3 - Simplified)

import os
import logging
import time
from aip import AipSpeech
import srt
from datetime import timedelta
import re
from moviepy.editor import AudioFileClip

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.client = AipSpeech(
            config.BAIDU_TTS_APP_ID,
            config.BAIDU_TTS_API_KEY,
            config.BAIDU_TTS_SECRET_KEY
        )
        self.client.setConnectionTimeoutInMillis(5000)
        self.client.setSocketTimeoutInMillis(15000)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.qps_limit_delay = 1.1
        self.max_retries = 3

    def _generate_audio_raw(self, text: str) -> bytes | None:
        """
        Calls the TTS API using the default voice options from config.
        """
        self.logger.info(f"Requesting TTS for text: '{text[:50]}...'")
        
        for attempt in range(self.max_retries):
            try:
                # 直接使用 config 中的 TTS_OPTIONS
                result = self.client.synthesis(text, 'zh', 1, self.config.TTS_OPTIONS)

                if isinstance(result, dict):
                    error_code = result.get('err_no')
                    error_msg = result.get('err_msg', 'Unknown API error')
                    self.logger.error(f"Baidu TTS API error. Code: {error_code}, Msg: {error_msg}. Attempt {attempt + 1}/{self.max_retries}.")
                    if error_code in [503, 3302, 3303]:
                        time.sleep(self.qps_limit_delay * (2 ** attempt))
                        continue
                    else:
                        return None
                
                if isinstance(result, bytes) and len(result) > 1024:
                    self.logger.info("Successfully received audio data from API.")
                    return result
                else:
                    self.logger.warning(f"Received unexpected/invalid data from TTS API (type: {type(result)}). Retrying...")
                    time.sleep(self.qps_limit_delay * (2 ** attempt))

            except Exception as e:
                self.logger.error(f"An unexpected exception in _generate_audio_raw: {e}", exc_info=True)
                time.sleep(self.qps_limit_delay * (2 ** attempt))
        
        self.logger.critical(f"Failed to generate raw audio after {self.max_retries} attempts.")
        return None
    
    def _create_srt_by_char_mapping(self, narration: str, audio_duration: float, srt_path: str):
        # This function remains unchanged
        narration_clean = re.sub(r'\s', '', narration)
        total_chars = len(narration_clean)
        if total_chars == 0:
            self.logger.warning("Narration is empty. Cannot create SRT.")
            open(srt_path, 'w').close()
            return
            
        char_duration = audio_duration / total_chars
        delimiters = "、，。！？"
        sentences = re.split(f'([{delimiters}])', narration)
        chunks = []
        temp_chunk = ""
        for part in sentences:
            if part:
                temp_chunk += part
                if part in delimiters or part == sentences[-1]:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = ""

        subtitles = []
        current_time = 0.0
        for i, chunk in enumerate(chunks):
            if not chunk: continue
            chunk_char_len = len(re.sub(r'\s', '', chunk))
            chunk_duration = chunk_char_len * char_duration
            start_time = timedelta(seconds=current_time)
            end_time = timedelta(seconds=min(current_time + chunk_duration, audio_duration))
            sub = srt.Subtitle(index=i+1, start=start_time, end=end_time, content=chunk)
            subtitles.append(sub)
            current_time += chunk_duration
        
        if subtitles:
            subtitles[-1].end = timedelta(seconds=audio_duration)

        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))
        self.logger.info(f"SRT file created by character mapping -> {srt_path}")

    def generate_audio(self, text: str, audio_output_path: str):
        """
        Generates audio with the default voice and creates corresponding SRT.
        The `gender` parameter is removed.
        """
        audio_data = self._generate_audio_raw(text)
        
        if not audio_data:
            self.logger.critical(f"Failed to generate audio for text '{text[:50]}...'.")
            return None, None
            
        try:
            with open(audio_output_path, 'wb') as f:
                f.write(audio_data)
            self.logger.info(f"Audio successfully saved to {audio_output_path}")
        except Exception as e:
            self.logger.error(f"Failed to write audio file {audio_output_path}: {e}")
            return None, None

        srt_output_path = os.path.splitext(audio_output_path)[0] + '.srt'
        try:
            with AudioFileClip(audio_output_path) as audio_clip:
                audio_duration = audio_clip.duration
            
            self._create_srt_by_char_mapping(text, audio_duration, srt_output_path)
            time.sleep(self.qps_limit_delay)
            return audio_output_path, srt_output_path

        except Exception as e:
            self.logger.error(f"Failed to get audio duration or create SRT file: {e}", exc_info=True)
            open(srt_output_path, 'w').close()
            return audio_output_path, None