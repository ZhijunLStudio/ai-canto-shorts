# src/final_assembler.py (V2.2 - Restored add_bgm method)

import logging
import os
import cv2
import subprocess
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip
)
from moviepy.video.fx.all import crop, resize

class FinalAssembler:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.FINAL_W, self.FINAL_H = 1080, 1920

    def _create_blur_composite(self, source_clip):
        kernel_size = (getattr(self.config, 'BLUR_KERNEL_SIZE', 51), getattr(self.config, 'BLUR_KERNEL_SIZE', 51))
        
        background_clip = resize(source_clip, height=self.FINAL_H)
        background_clip = background_clip.fl_image(lambda frame: cv2.GaussianBlur(frame, kernel_size, 0))
        background_clip = crop(
            background_clip, 
            width=self.FINAL_W, height=self.FINAL_H, 
            x_center=background_clip.w/2, y_center=background_clip.h/2
        )
        foreground_clip = resize(source_clip, width=self.FINAL_W)
        foreground_clip = foreground_clip.set_position(('center', 'center'))
        return CompositeVideoClip([background_clip, foreground_clip], size=(self.FINAL_W, self.FINAL_H))

    def combine_video_audio(self, style, audio_path, output_path, segment_info, original_video_path, camera_path=None):
        all_clips = []
        try:
            audio_clip = AudioFileClip(audio_path)
            all_clips.append(audio_clip)
            target_duration = audio_clip.duration

            original_clip_obj = VideoFileClip(original_video_path, audio=False)
            all_clips.append(original_clip_obj)
            start_t = segment_info['start_time']
            end_t = segment_info['end_time'] 
            
            if end_t > original_clip_obj.duration:
                self.logger.warning(f"Corrected end time ({end_t:.2f}s) exceeds source video duration ({original_clip_obj.duration:.2f}s). Trimming.")
                end_t = original_clip_obj.duration
                audio_clip = audio_clip.subclip(0, end_t - start_t)
            
            source_subclip = original_clip_obj.subclip(start_t, end_t)
            all_clips.append(source_subclip)
            
            final_visual_clip = None
            if style == 'BLUR_COMPOSITE':
                final_visual_clip = self._create_blur_composite(source_subclip)
            elif style == 'CROP_VERTICAL':
                if not camera_path:
                    self.logger.error("For 'CROP_VERTICAL' style, camera_path is missing.")
                    return None
                
                def dynamic_crop_and_resize(get_frame, t):
                    frame = get_frame(t)
                    h, w = frame.shape[:2]
                    crop_w = int(h * (self.FINAL_W / self.FINAL_H))
                    
                    frame_index_in_path = int(t * source_subclip.fps)
                    if frame_index_in_path >= len(camera_path):
                        frame_index_in_path = len(camera_path) - 1

                    center_x = camera_path[frame_index_in_path]
                    
                    x1 = max(0, center_x - crop_w / 2.0)
                    x2 = min(w, x1 + crop_w)
                    x1 = x2 - crop_w

                    cropped_frame = frame[:, int(x1):int(x2)]
                    return cv2.resize(cropped_frame, (self.FINAL_W, self.FINAL_H), interpolation=cv2.INTER_AREA)
                
                final_visual_clip = source_subclip.fl(dynamic_crop_and_resize)
                final_visual_clip.size = (self.FINAL_W, self.FINAL_H)
            else:
                self.logger.error(f"Unknown video style requested: '{style}'.")
                return None
            
            all_clips.append(final_visual_clip)
            final_clip_with_audio = final_visual_clip.set_audio(audio_clip)
            all_clips.append(final_clip_with_audio)

            final_clip_with_audio.write_videofile(
                output_path, codec='libx264', audio_codec='aac', 
                preset='veryfast', logger=None, threads=4,
                ffmpeg_params=['-s', f'{self.FINAL_W}x{self.FINAL_H}']
            )
            self.logger.info(f"Successfully created '{style}' sub-clip: {os.path.basename(output_path)}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to combine video for style '{style}': {e}", exc_info=True)
            return None
        finally:
            for clip in all_clips:
                if clip:
                    try: clip.close()
                    except Exception: pass

    def stitch_clips(self, clip_paths, final_output_path):
        if not clip_paths:
            self.logger.error("No clips to stitch.")
            return None
        
        self.logger.info(f"Stitching {len(clip_paths)} clips using FFMPEG concat filter (re-encoding)...")
        
        inputs = " ".join([f"-i \"{os.path.abspath(path)}\"" for path in clip_paths])
        filter_str = "".join([f"[{i}:v][{i}:a]" for i in range(len(clip_paths))])
        filter_str += f"concat=n={len(clip_paths)}:v=1:a=1[outv][outa]"
        command = (
            f"ffmpeg -y {inputs} -filter_complex \"{filter_str}\" "
            f"-map \"[outv]\" -map \"[outa]\" -c:v libx264 -preset veryfast "
            f"-c:a aac -ar 44100 \"{final_output_path}\""
        )
        
        try:
            self.logger.debug(f"Executing FFMPEG stitch command:\n{command}")
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
            self.logger.info(f"🎉 Final video stitched successfully: {os.path.basename(final_output_path)}")
            return final_output_path
        except FileNotFoundError:
            self.logger.error("FFMPEG command not found. Please ensure FFMPEG is installed and in your system's PATH.")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFMPEG failed during stitching. Stderr:\n{e.stderr}")
            return None

    # [NEW] Restored add_bgm method from a previous version.
    def add_bgm(self, video_path, bgm_path, output_path, bgm_volume=0.4):
        self.logger.info(f"Adding BGM to '{os.path.basename(video_path)}'...")
        temp_files = []
        try:
            output_dir = os.path.dirname(output_path)
            temp_narration_audio = os.path.join(output_dir, "temp_narration.aac")
            temp_bgm_converted = os.path.join(output_dir, "temp_bgm_converted.mp3")
            temp_final_audio = os.path.join(output_dir, "temp_final_mixed_audio.aac")
            temp_files.extend([temp_narration_audio, temp_bgm_converted, temp_final_audio])

            # Extract narration audio from the stitched video
            subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'copy', temp_narration_audio], check=True, capture_output=True)
            
            # Convert BGM to a standard format to ensure compatibility
            subprocess.run(['ffmpeg', '-y', '-i', bgm_path, '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', temp_bgm_converted], check=True, capture_output=True)
            
            # Mix narration and BGM
            cmd_mix = [
                'ffmpeg', '-y', '-i', temp_narration_audio, '-i', temp_bgm_converted,
                '-filter_complex', f"[0:a]volume=1.0[a0];[1:a]volume={bgm_volume}[a1];[a0][a1]amix=inputs=2:duration=first[aout]",
                '-map', '[aout]', temp_final_audio
            ]
            subprocess.run(cmd_mix, check=True, capture_output=True)
            
            # Merge the new mixed audio with the original video stream
            cmd_merge = [
                'ffmpeg', '-y', '-i', video_path, '-i', temp_final_audio,
                '-c:v', 'copy', '-map', '0:v:0', '-c:a', 'aac', '-map', '1:a:0', '-shortest', output_path
            ]
            subprocess.run(cmd_merge, check=True, capture_output=True)
            
            self.logger.info(f"🎉 Successfully created final video with BGM: {os.path.basename(output_path)}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"An FFMPEG command failed during BGM addition. Stderr:\n" + e.stderr.decode('utf-8', errors='ignore'))
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during BGM addition: {e}", exc_info=True)
            return None
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temp file {f}: {e}")