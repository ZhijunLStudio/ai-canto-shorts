# app.py (V3.7 - UI Enhancements & Plan Saving)

import gradio as gr
import os, sys, shutil, logging, time
from datetime import datetime
import subprocess, json
from collections import defaultdict
import cv2
import srt
from datetime import timedelta
from moviepy.editor import VideoFileClip

# --- Add src to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- Import project modules ---
from src.pre_analyzer import VideoAndAudioAnalyzer
from src.video_processor import SmartVideoProcessor
from src.audio_processor import AudioProcessor
from src.final_assembler import FinalAssembler
from src.sub_fixer import SubtitleFixer
from src.data_manager import data_manager # Import the global instance
from src.main_workflow import VideoToShortsWorkflow
import config
from aip import AipSpeech
from openai import OpenAI
from moviepy.editor import AudioFileClip









def generate_clipping_plan():
    """
    运行 Agent 工作流，通过大模型分析视频，生成一份详细的剪辑计划 (JSON 格式)。
    """
    logger = logging.getLogger("Phase1_PlanGenerator")
    logger.info("启动视频剪辑 Agent 工作流...")

    if not data_manager.is_initialized:
        logger.critical("DataManager 未初始化！无法运行工作流。")
        return None

    workflow = VideoToShortsWorkflow()
    response_iterator = workflow.run()
    
    print("\n" + "="*50)
    print("🚀 阶段一: 正在生成剪辑计划...")
    print("="*50)
    for response in response_iterator:
        content = response.content if hasattr(response, 'content') else str(response)
        if not content.strip().startswith('{'):
            print(content.strip())

    logger.info("Agent 工作流执行完毕。")
    print("\n" + "="*50)
    print("✅ 阶段一完成: 剪辑计划已生成 🎉")
    print("="*50)

    final_plan = workflow.session_state.get("final_output")

    if final_plan:
        print(json.dumps(final_plan, indent=2, ensure_ascii=False))
        logger.info("成功生成剪辑方案。")
        return final_plan
    else:
        print("工作流未能成功生成最终剪辑计划。")
        print("请检查上面的日志，查找 '工作流因错误而终止' 的信息。")
        logger.warning("工作流未生成最终产出 'final_output'。")
        return None

# ==============================================================================
# ---                     辅助函数：拼接 SRT 文件                        ---
# ==============================================================================
def stitch_srt_files_with_offset(video_clip_paths: list, srt_file_paths: list, output_srt_path: str):
    """
    根据视频片段的实际时长，精确地拼接SRT文件。
    """
    logger = logging.getLogger("SRT_Stitcher")
    if len(video_clip_paths) != len(srt_file_paths):
        logger.error(f"视频片段数量({len(video_clip_paths)})和SRT文件数量({len(srt_file_paths)})不匹配，无法拼接。")
        return False
        
    all_subs = []
    master_index = 1
    current_time_offset = timedelta(0)

    for i, (video_path, srt_path) in enumerate(zip(video_clip_paths, srt_file_paths)):
        try:
            with VideoFileClip(video_path) as clip:
                clip_duration = timedelta(seconds=clip.duration)

            with open(srt_path, 'r', encoding='utf-8') as f:
                subs_for_clip = list(srt.parse(f.read()))
            
            for sub in subs_for_clip:
                sub.index = master_index
                sub.start += current_time_offset
                sub.end += current_time_offset
                all_subs.append(sub)
                master_index += 1
            
            current_time_offset += clip_duration

        except FileNotFoundError:
            logger.error(f"文件未找到: {srt_path} 或 {video_path}。跳过此片段的拼接。")
            continue
        except Exception as e:
            logger.error(f"处理文件 {srt_path} 时出错: {e}")
            continue
            
    try:
        final_srt_content = srt.compose(all_subs)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(final_srt_content)
        logger.info(f"SRT文件成功拼接 -> {output_srt_path}")
        return True
    except Exception as e:
        logger.error(f"写入最终SRT文件 {output_srt_path} 时出错: {e}")
        return False

# ==============================================================================
# ---                     Backend Logic Wrapper Class                        ---
# ==============================================================================
class GradioAppLogic:
    def __init__(self):
        self.run_base_dir, self.clipping_plan = None, None
        self.stitched_videos, self.stitched_srts = {}, {}
        self.downloaded_bgm_path, self.narration_audio_path = None, None
        self.base_filename = "final_video"
        self.logger = logging.getLogger("GradioApp")

    def setup_environment_and_logging(self, video_path):
        if video_path and os.path.exists(video_path): self.base_filename = os.path.splitext(os.path.basename(video_path))[0]
        else: self.base_filename = "loaded_json_run"
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.run_base_dir = os.path.join(config.BASE_OUTPUT_DIR, f"{timestamp}_{self.base_filename}")
        config.RUN_BASE_DIR, config.VIDEO_PATH = self.run_base_dir, video_path
        
        for subdir_key in ['SILENT_CLIPS_DIR', 'AUDIO_FILES_DIR', 'FINAL_CLIPS_DIR', 'DEBUG_VIDEOS_DIR', 'BGM_DIR']:
            dir_name = subdir_key.split('_DIR')[0].lower().replace('_', '-')
            path = os.path.join(self.run_base_dir, dir_name)
            setattr(config, subdir_key, path); os.makedirs(path, exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if root_logger.hasHandlers(): root_logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(formatter); root_logger.addHandler(console_handler)
        log_file_path = os.path.join(self.run_base_dir, 'run_log.log')
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8'); file_handler.setFormatter(formatter); root_logger.addHandler(file_handler)
        
        self.logger.info(f"🚀 环境准备完成。日志将同时输出到终端和文件: {log_file_path}")
        return self.run_base_dir

    def run_plan_generation(self, video_json_path, audio_json_path):
        from src.data_manager import data_manager
        data_manager.__init__(); data_manager.initialize_from_files(video_json_path, audio_json_path, skip_seconds=getattr(config, 'SKIP_FIRST_N_SECONDS', 0))
        self.clipping_plan = generate_clipping_plan()
        
        if not self.clipping_plan or "clips" not in self.clipping_plan:
             self.logger.error("未能生成有效的剪辑计划。")
             return False
        
        # [NEW] Save the generated plan to the run directory
        plan_save_path = os.path.join(self.run_base_dir, "final_plan.json")
        try:
            with open(plan_save_path, 'w', encoding='utf-8') as f:
                json.dump(self.clipping_plan, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✅ 剪辑计划已成功生成并保存至: {plan_save_path}")
        except Exception as e:
            self.logger.error(f"保存剪辑计划失败: {e}")

        return True

    def run_video_assembly(self):
        # ... (This entire method remains unchanged from the "Audio First" version)
        self.logger.info("\n" + "="*50 + "\n🎬 STEP 2: Video and Audio Synthesis\n" + "="*50)
        video_processor = SmartVideoProcessor(config)
        audio_processor = AudioProcessor(config)
        assembler = FinalAssembler(config)
        sub_fixer = SubtitleFixer(remove_punctuation=True)
        initial_clips = self.clipping_plan.get("clips")
        if not initial_clips: self.logger.error("No 'clips' found in the plan. Aborting."); return None, None, None, None
        self.logger.info("\n--- STAGE 2.1: Generating Audio and Finalizing Timings ---")
        clips_with_final_timing, audio_assets, full_narration_text_list = [], {}, []
        for i, seg_info in enumerate(initial_clips):
            if 'start_time' not in seg_info or 'narration' not in seg_info: self.logger.warning(f"Segment {i+1} is malformed. Skipping. Data: {seg_info}"); continue
            narration = seg_info['narration'].strip()
            if not narration: self.logger.warning(f"Segment {i+1} has empty narration. Skipping."); continue
            self.logger.info(f"Processing audio for segment {i+1}/{len(initial_clips)}...")
            clip_id = f"clip_{i}"
            audio_path, srt_path = audio_processor.generate_audio(narration, os.path.join(config.AUDIO_FILES_DIR, f"audio_{clip_id}.mp3"))
            if not audio_path: self.logger.error(f"Failed to generate audio for segment {i+1}. Skipping."); continue
            try:
                with AudioFileClip(audio_path) as audio_clip: actual_duration = audio_clip.duration
            except Exception as e: self.logger.error(f"Could not read duration from {audio_path}: {e}. Skipping."); continue
            updated_seg_info = seg_info.copy()
            updated_seg_info['end_time'] = updated_seg_info['start_time'] + actual_duration
            clips_with_final_timing.append(updated_seg_info)
            audio_assets[clip_id] = {'audio_path': audio_path, 'srt_path': srt_path}
            full_narration_text_list.append(narration)
            self.logger.info(f"  -> Audio generated. Actual duration: {actual_duration:.2f}s.")
        self.logger.info("\n--- STAGE 2.2: Processing Video based on Final Timings ---")
        if not clips_with_final_timing: self.logger.error("No valid clips after audio processing. Aborting."); return None, None, None, None
        pose_data = video_processor._analyze_video(config.VIDEO_PATH, clips_with_final_timing)
        fps = cv2.VideoCapture(config.VIDEO_PATH).get(cv2.CAP_PROP_FPS)
        final_segments, original_indices = video_processor._preprocess_segments(clips_with_final_timing, pose_data, fps)
        camera_paths_dict, processed_segments = video_processor._calculate_camera_paths(config.VIDEO_PATH, final_segments, pose_data, original_indices)
        if config.CREATE_DEBUG_VIDEO and camera_paths_dict: video_processor._create_debug_video(config.VIDEO_PATH, processed_segments, pose_data, camera_paths_dict, original_indices)
        self.logger.info("\n--- STAGE 2.3: Assembling Final Video Clips ---")
        final_clips_by_style, srt_files_by_style = defaultdict(list), defaultdict(list)
        for i, seg_info in enumerate(processed_segments):
            clip_id = f"clip_{original_indices[i]}"
            assets = audio_assets.get(clip_id)
            if not assets: continue
            for style in config.VIDEO_OUTPUT_STYLES:
                fixed_srt = os.path.join(config.AUDIO_FILES_DIR, f"fixed_srt_{style}_{clip_id}.srt")
                if not sub_fixer.process_srt_file(assets['srt_path'], fixed_srt): continue
                cam_path = camera_paths_dict.get(f"segment_{original_indices[i]}") if style == 'CROP_VERTICAL' else None
                final_clip_path = os.path.join(config.FINAL_CLIPS_DIR, f"final_clip_{style}_{clip_id}.mp4")
                combined_path = assembler.combine_video_audio(style, assets['audio_path'], final_clip_path, seg_info, config.VIDEO_PATH, cam_path)
                if combined_path:
                    final_clips_by_style[style].append(combined_path)
                    if os.path.exists(fixed_srt): srt_files_by_style[style].append(fixed_srt)
        for style, clips in final_clips_by_style.items():
            if not clips: continue
            narration_video_path = os.path.join(self.run_base_dir, f"{self.base_filename}_{style}_narration.mp4")
            stitched_video = assembler.stitch_clips(clips, narration_video_path)
            if stitched_video: self.stitched_videos[style] = stitched_video
            srts_to_stitch = srt_files_by_style.get(style, [])
            if srts_to_stitch:
                srt_out_path = os.path.join(self.run_base_dir, f"subtitles_{style}.srt")
                if stitch_srt_files_with_offset(clips, srts_to_stitch, srt_out_path): self.stitched_srts[style] = srt_out_path
        full_narration_text = " ".join(full_narration_text_list)
        if full_narration_text:
             self.narration_audio_path, _ = audio_processor.generate_audio(full_narration_text, os.path.join(self.run_base_dir, "full_narration.mp3"))
        bgm_keywords_str = self.clipping_plan.get("bgm_keywords")
        try:
            from src.bgm_downloader import download_first_successful_song
            if bgm_keywords_str: self.downloaded_bgm_path = download_first_successful_song(keyword=bgm_keywords_str, outdir=config.BGM_DIR, try_count=3)
        except ImportError: self.logger.warning("BGM下载器模块 'src/bgm_downloader.py' 未找到，跳过BGM下载。")
        return self.stitched_videos, self.narration_audio_path, bgm_keywords_str, self.downloaded_bgm_path

    def finalize_with_bgm(self, bgm_path):
        # ... (This entire method remains unchanged)
        assembler = FinalAssembler(config)
        font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fonts', '霞鹜文楷bold.ttf'))
        burn_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'burn_subtitles.sh'))
        if not os.path.exists(font_path) or not os.path.exists(burn_script_path): return []
        title = self.clipping_plan.get('title', '默认标题')
        title_formatted = title.replace('，', '\\N').replace('。', '\\N').replace('？', '\\N').replace('！', '\\N')
        final_paths = []
        for style, video_path in self.stitched_videos.items():
            srt_path = self.stitched_srts.get(style, "")
            self.logger.info(f"-> 正在为 '{style}' 生成 [无BGM, 带字幕] 版本...")
            output_no_bgm = os.path.join(self.run_base_dir, f"{self.base_filename}_{style}_noBGM_final.mp4")
            cmd_no_bgm = ['bash', burn_script_path, video_path, srt_path, output_no_bgm, title_formatted, font_path]
            subprocess.run(cmd_no_bgm, check=True, capture_output=True, text=True, encoding='utf-8')
            final_paths.append(output_no_bgm)
            self.logger.info(f"  -> 成功生成: {os.path.basename(output_no_bgm)}")
            if bgm_path and os.path.exists(bgm_path):
                self.logger.info(f"-> 正在为 '{style}' 生成 [带BGM, 带字幕] 版本...")
                bgm_video_path = assembler.add_bgm(video_path, bgm_path, os.path.join(self.run_base_dir, f"temp_{style}_with_bgm.mp4"))
                if bgm_video_path:
                    output_with_bgm = os.path.join(self.run_base_dir, f"{self.base_filename}_{style}_withBGM_final.mp4")
                    cmd_with_bgm = ['bash', burn_script_path, bgm_video_path, srt_path, output_with_bgm, title_formatted, font_path]
                    subprocess.run(cmd_with_bgm, check=True, capture_output=True, text=True, encoding='utf-8')
                    final_paths.append(output_with_bgm)
                    self.logger.info(f"  -> 成功生成: {os.path.basename(output_with_bgm)}")
        self.logger.info("\n🎉 所有最终视频生成完毕！")
        return final_paths

# ==============================================================================
# ---                        Gradio UI Helper & Main Flow                      ---
# ==============================================================================
def test_baidu_llm_connection(api_key, base_url):
    # ... (no changes)
    if not all([api_key, base_url]): return "❌ 错误: 请填写API密钥和基础URL。"
    try: client = OpenAI(api_key=api_key, base_url=base_url); client.models.list(); return "✅ 成功: 连接正常。"
    except Exception as e: return f"❌ 失败: {str(e)}"

def test_baidu_tts_connection(app_id, api_key, secret_key):
    # ... (no changes)
    if not all([app_id, api_key, secret_key]): return "❌ 错误: 请填写所有TTS凭证。"
    try:
        client = AipSpeech(app_id, api_key, secret_key)
        result = client.synthesis('测', 'zh', 1, {'vol': 5, 'per': 0})
        if isinstance(result, dict) and 'err_no' in result: return f"❌ 失败: API返回 - {result.get('err_msg', '未知错误')}"
        return "✅ 成功: 连接正常。"
    except Exception as e: return f"❌ 失败: {str(e)}"

def process_video_flow(
    video_path, model_path, 
    video_llm_api_key, video_llm_base_url, video_llm_model_name,
    agent_llm_api_key, agent_llm_base_url, agent_llm_model_name,
    tts_app_id, tts_api_key, tts_secret_key, 
    analysis_mode, video_json_file, audio_json_file,
    run_mode):
    
    # ... (no changes in initial part) ...
    WORKFLOW_STAGES = [
        (1, "环境准备"), (2, "音频与视频内容分析"), (3, "AI 智能编排"),
        (4, "视频与音频片段合成"), (5, "最终视频渲染")
    ]
    TOTAL_STAGES = len(WORKFLOW_STAGES)
    def get_stage_text(stage_index, note=""):
        if stage_index >= len(WORKFLOW_STAGES): return "**所有步骤已完成**"
        stage_num, stage_name = WORKFLOW_STAGES[stage_index]
        anim_chars = ["-", "\\", "|", "/"]
        anim = anim_chars[int(time.time()*2) % 4]
        return f"**进度: {stage_num}/{TOTAL_STAGES} - {stage_name} {anim}**\n_{note}_"
    yield { status_textbox: get_stage_text(0, "正在初始化..."), manual_mode_controls: gr.update(visible=False), results_group: gr.update(visible=False), start_button: gr.update(interactive=False) }
    if analysis_mode == "从头生成分析文件" and not video_path:
        yield {status_textbox: "❌ **错误**: 请上传一个视频文件以便进行新分析。", start_button: gr.update(interactive=True)}; return
    if analysis_mode == "加载已有分析文件" and (not video_json_file or not audio_json_file or not video_path):
        yield {status_textbox: "❌ **错误**: 加载JSON时，必须同时提供原始视频文件、视频JSON和音频JSON。", start_button: gr.update(interactive=True)}; return
    config.SKIP_FIRST_N_SECONDS = getattr(config, 'SKIP_FIRST_N_SECONDS', 0)
    config.MODEL_PATH, config.VIDEO_LLM_API_KEY, config.VIDEO_LLM_BASE_URL, config.VIDEO_LLM_MODEL_NAME = model_path, video_llm_api_key, video_llm_base_url, video_llm_model_name
    config.AGENT_LLM_API_KEY, config.AGENT_LLM_BASE_URL, config.AGENT_LLM_MODEL_NAME = agent_llm_api_key, agent_llm_base_url, agent_llm_model_name
    config.BAIDU_TTS_APP_ID, config.BAIDU_TTS_API_KEY, config.BAIDU_TTS_SECRET_KEY = tts_app_id, tts_api_key, tts_secret_key
    logic = GradioAppLogic()
    run_dir = logic.setup_environment_and_logging(video_path)
    
    final_video_json_path, final_audio_json_path = None, None
    if analysis_mode == "从头生成分析文件":
        analyzer = VideoAndAudioAnalyzer()
        yield {status_textbox: get_stage_text(1, "正在识别语音...")}
        audio_content = analyzer.analyze_audio_content(config.VIDEO_PATH)
        yield {status_textbox: get_stage_text(1, "使用Vision-LLM逐帧分析，此步耗时较长...")}
        video_content = analyzer.analyze_video_content(config.VIDEO_PATH)
        if not video_content or not audio_content:
            yield {status_textbox: "❌ **错误**: 视频或音频分析失败，请检查终端日志。", start_button: gr.update(interactive=True)}; return
        final_video_json_path, final_audio_json_path = os.path.join(run_dir, "analysis_video.json"), os.path.join(run_dir, "analysis_audio.json")
        with open(final_video_json_path, 'w', encoding='utf-8') as f: json.dump(video_content, f, indent=2, ensure_ascii=False)
        with open(final_audio_json_path, 'w', encoding='utf-8') as f: json.dump(audio_content, f, indent=2, ensure_ascii=False)
    else:
        yield {status_textbox: get_stage_text(1, "跳过分析，正在加载JSON文件...")}; time.sleep(1)
        final_video_json_path = shutil.copy(video_json_file.name, os.path.join(run_dir, "analysis_video.json"))
        final_audio_json_path = shutil.copy(audio_json_file.name, os.path.join(run_dir, "analysis_audio.json"))
        
    yield {status_textbox: get_stage_text(2, "AI Agent正在进行多轮头脑风暴...")}
    plan_success = logic.run_plan_generation(final_video_json_path, final_audio_json_path)
    if not plan_success:
        yield {status_textbox: "❌ **错误**: Agent工作流失败，请检查终端日志。", start_button: gr.update(interactive=True)}; return

    yield {status_textbox: get_stage_text(3, "正在处理视频片段和生成音频...")}
    stitched_videos, narration_audio, bgm_keywords, downloaded_bgm = logic.run_video_assembly()
    if not stitched_videos:
        yield {status_textbox: "❌ **错误**: 视频组装失败，请检查终端日志。", start_button: gr.update(interactive=True)}; return

    # [MODIFIED] Centralized UI update logic for the end of the process
    def final_ui_update(final_videos, logic_instance):
        plan = logic_instance.clipping_plan
        title = plan.get('title', 'N/A')
        keywords = ", ".join(plan.get('keywords', []))
        description = plan.get('description', 'N/A')
        
        video_outputs = { 
            final_video_1: gr.update(visible=False), 
            final_video_2: gr.update(visible=False) 
        }
        if len(final_videos) > 0: video_outputs[final_video_1] = gr.update(value=final_videos[0], visible=True, label=os.path.basename(final_videos[0]))
        if len(final_videos) > 1: video_outputs[final_video_2] = gr.update(value=final_videos[1], visible=True, label=os.path.basename(final_videos[1]))
        
        return {
            **video_outputs,
            status_textbox: f"✅ **流程完成！**\n所有文件已保存在: `{logic_instance.run_base_dir}`",
            start_button: gr.update(interactive=True),
            results_group: gr.update(visible=True),
            result_title: gr.update(value=title),
            result_keywords: gr.update(value=keywords),
            result_description: gr.update(value=description),
            app_state: None # Clear state
        }

    if run_mode == "全自动模式":
        yield {status_textbox: get_stage_text(4, "正在合成最终视频...")}
        final_videos = logic.finalize_with_bgm(downloaded_bgm)
        yield final_ui_update(final_videos, logic)
    else:
        yield { 
            status_textbox: f"⏸️ **手动模式**: 流程暂停，请审核旁白并选择BGM。", 
            manual_mode_controls: gr.update(visible=True), 
            narration_audio_player: gr.update(value=narration_audio), 
            bgm_suggestion_text: gr.update(value=f"AI推荐关键词: '{bgm_keywords}'"), 
            ai_bgm_player: gr.update(value=downloaded_bgm), 
            app_state: logic 
        }

def finalize_manual_mode(logic_instance, custom_bgm_file):
    # ... (This logic remains largely the same but uses the new final_ui_update helper)
    if not logic_instance:
        return {status_textbox: "❌ **错误**: 应用状态丢失，请重新开始。"}
    
    yield { status_textbox: "**进度: 5/5 - 最终视频渲染 ⏳**\n_正在使用您选择的BGM进行最后合成..._", manual_mode_controls: gr.update(visible=False), start_button: gr.update(interactive=False) }

    chosen_bgm = None
    if custom_bgm_file:
        chosen_bgm = custom_bgm_file.name
        logic_instance.logger.info(f"手动模式确认，使用用户上传的BGM: {os.path.basename(chosen_bgm)}")
    elif logic_instance.downloaded_bgm_path:
        chosen_bgm = logic_instance.downloaded_bgm_path
        logic_instance.logger.info(f"手动模式确认，使用AI下载的BGM: {os.path.basename(chosen_bgm)}")
    else:
        logic_instance.logger.info("手动模式确认，无BGM。")

    final_videos = logic_instance.finalize_with_bgm(chosen_bgm)
    
    # Re-use the final UI update function
    def final_ui_update(final_videos, logic_instance):
        plan = logic_instance.clipping_plan
        title = plan.get('title', 'N/A')
        keywords = ", ".join(plan.get('keywords', []))
        description = plan.get('description', 'N/A')
        
        video_outputs = { 
            final_video_1: gr.update(visible=False), 
            final_video_2: gr.update(visible=False) 
        }
        if len(final_videos) > 0: video_outputs[final_video_1] = gr.update(value=final_videos[0], visible=True, label=os.path.basename(final_videos[0]))
        if len(final_videos) > 1: video_outputs[final_video_2] = gr.update(value=final_videos[1], visible=True, label=os.path.basename(final_videos[1]))
        
        return {
            **video_outputs,
            status_textbox: f"✅ **流程完成！**\n所有文件已保存在: `{logic_instance.run_base_dir}`",
            start_button: gr.update(interactive=True),
            results_group: gr.update(visible=True),
            result_title: gr.update(value=title),
            result_keywords: gr.update(value=keywords),
            result_description: gr.update(value=description),
            app_state: None,
            manual_mode_controls: gr.update(visible=False)
        }
    
    yield final_ui_update(final_videos, logic_instance)


# ==============================================================================
# ---                            Gradio UI Layout (V3.7)                       ---
# ==============================================================================
css = ".gradio-container { font-family: 'IBM Plex Sans', sans-serif; } footer { display: none !important; } .gr-prose h2 { letter-spacing: -0.05em; font-weight: 600; } .gr-prose p { margin-bottom: 0.5rem; }"
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=css) as demo:
    app_state = gr.State(None)
    gr.Markdown("# 🎬 AI短视频全自动生成器 v3.7")
    gr.Markdown("一个端到端的自动化工作流：从长视频分析、AI编导、角色扮演式配音，到最终生成带BGM和字幕的短视频。")

    with gr.Tabs():
        with gr.TabItem("⚙️ 参数配置"):
            # ... (no changes to the config tab layout)
            gr.Markdown("### 在开始前，请检查并配置您的模型和API凭证。")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 🚶‍♂️ 姿态识别模型")
                    gr.Markdown("<p style='color:#666; font-size:small;'>用于智能运镜，识别视频中的人物动作。请提供本地YOLOv8 Pose模型的路径。</p>")
                    with gr.Accordion("模型路径", open=True):
                        model_path_input = gr.Textbox(label="YOLO 姿态模型路径", value=config.MODEL_PATH, placeholder="例如: yolov8n-pose.pt")
                    gr.Markdown("## 🎙️ 语音合成 (旁白)")
                    gr.Markdown("<p style='color:#666; font-size:small;'>用于将AI生成的旁白文本转换为语音。推荐使用百度TTS。</p>")
                    with gr.Accordion("百度TTS凭证", open=True):
                        tts_app_id_input = gr.Textbox(label="TTS App ID", type="password", value=config.BAIDU_TTS_APP_ID)
                        tts_api_key_input = gr.Textbox(label="TTS API Key", type="password", value=config.BAIDU_TTS_API_KEY)
                        tts_secret_key_input = gr.Textbox(label="TTS Secret Key", type="password", value=config.BAIDU_TTS_SECRET_KEY)
                        with gr.Row():
                            test_tts_button = gr.Button("测试TTS连接")
                            tts_status_textbox = gr.Textbox(label="连接状态", interactive=False)
                with gr.Column(scale=2):
                    gr.Markdown("## 🧠 大语言模型 (核心大脑)")
                    gr.Markdown("<p style='color:#666; font-size:small;'>整个工作流的核心，负责视频理解和内容创作。支持兼容OpenAI API接口的任何模型服务。</p>")
                    with gr.Accordion("视频分析大模型", open=True):
                        gr.Markdown("**多模态模型**：用于从视频帧中提取视觉信息。")
                        video_llm_api_key_input = gr.Textbox(label="API Key", type="password", value=config.VIDEO_LLM_API_KEY)
                        video_llm_base_url_input = gr.Textbox(label="Base URL", value=config.VIDEO_LLM_BASE_URL)
                        video_llm_model_name_input = gr.Dropdown(label="模型名称", choices=["ernie-4.5-vl-28b-a3b", "ernie-4.5-turbo-vl-preview"], value=config.VIDEO_LLM_MODEL_NAME, allow_custom_value=True)
                        with gr.Row():
                            test_video_llm_button = gr.Button("测试连接")
                            video_llm_status_textbox = gr.Textbox(label="连接状态", interactive=False)
                    with gr.Accordion("Agent工作流大模型", open=True):
                        gr.Markdown("**文本模型**：用于剪辑规划、旁白创作等Agent任务。")
                        agent_llm_api_key_input = gr.Textbox(label="API Key", type="password", value=config.AGENT_LLM_API_KEY)
                        agent_llm_base_url_input = gr.Textbox(label="Base URL", value=config.AGENT_LLM_BASE_URL)
                        agent_llm_model_name_input = gr.Dropdown(label="文本模型", choices=["ernie-4.5-turbo-128k-preview", "ernie-4.5-turbo-vl-preview"], value=config.AGENT_LLM_MODEL_NAME, allow_custom_value=True)
                        with gr.Row():
                            test_agent_llm_button = gr.Button("测试连接")
                            agent_llm_status_textbox = gr.Textbox(label="连接状态", interactive=False)

        with gr.TabItem("🚀 主工作流"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 输入源")
                    analysis_mode_radio = gr.Radio(["从头生成分析文件", "加载已有分析文件"], label="分析模式", value="从头生成分析文件")
                    video_input = gr.Video(label="上传源视频", visible=True)
                    with gr.Group(visible=False) as load_json_group:
                        gr.Markdown("上传预分析的JSON文件，**同时仍需上传原始视频**用于画面裁切。")
                        video_json_upload = gr.File(label="上传 analysis_video.json")
                        audio_json_upload = gr.File(label="上传 analysis_audio.json")
                    
                    gr.Markdown("### 2. 运行模式")
                    run_mode_selection = gr.Radio(["全自动模式", "手动审核模式"], label="选择模式", value="全自动模式", info="全自动: 一键到底。\n手动审核: 在BGM合成前暂停，供您审核。")
                    start_button = gr.Button("🚀 开始处理", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("### 3. 运行状态与结果")
                    status_textbox = gr.Markdown(value="**状态**: 准备就绪")
                    with gr.Group(visible=False) as manual_mode_controls:
                        gr.Markdown("### ⏸️ 手动审核面板")
                        gr.Markdown("请试听旁白，并确认/上传背景音乐。")
                        narration_audio_player = gr.Audio(label="试听完整旁白", type="filepath")
                        with gr.Row():
                            with gr.Column():
                                bgm_suggestion_text = gr.Textbox(label="BGM推荐关键词", interactive=False)
                                ai_bgm_player = gr.Audio(label="试听AI下载的BGM", type="filepath")
                            with gr.Column():
                                custom_bgm_upload = gr.File(label="... 或上传您自己的BGM")
                        finalize_button = gr.Button("✅ 确认并继续合成", variant="primary")
                    
                    # [NEW] Results group to display text outputs
                    with gr.Group(visible=False) as results_group:
                        gr.Markdown("### 📜 AI生成文案")
                        result_title = gr.Textbox(label="标题", interactive=False)
                        result_keywords = gr.Textbox(label="关键词", interactive=False)
                        result_description = gr.Textbox(label="简介", interactive=False, lines=3)
                        with gr.Row():
                            final_video_1 = gr.Video(label="最终视频产出 1", visible=False)
                            final_video_2 = gr.Video(label="最终视频产出 2", visible=False)

    # --- Event Handlers ---
    def toggle_analysis_inputs(mode):
        return {load_json_group: gr.update(visible=(mode == "加载已有分析文件"))}
    
    analysis_mode_radio.change(fn=toggle_analysis_inputs, inputs=analysis_mode_radio, outputs=[load_json_group])
    
    test_video_llm_button.click(fn=test_baidu_llm_connection, inputs=[video_llm_api_key_input, video_llm_base_url_input], outputs=video_llm_status_textbox)
    test_agent_llm_button.click(fn=test_baidu_llm_connection, inputs=[agent_llm_api_key_input, agent_llm_base_url_input], outputs=agent_llm_status_textbox)
    test_tts_button.click(fn=test_baidu_tts_connection, inputs=[tts_app_id_input, tts_api_key_input, tts_secret_key_input], outputs=tts_status_textbox)
    
    # [MODIFIED] Update the outputs list for start_button.click
    start_button.click(
        fn=process_video_flow,
        inputs=[
            video_input, model_path_input, 
            video_llm_api_key_input, video_llm_base_url_input, video_llm_model_name_input,
            agent_llm_api_key_input, agent_llm_base_url_input, agent_llm_model_name_input,
            tts_app_id_input, tts_api_key_input, tts_secret_key_input,
            analysis_mode_radio, video_json_upload, audio_json_upload,
            run_mode_selection
        ],
        outputs=[
            status_textbox, manual_mode_controls, narration_audio_player, 
            bgm_suggestion_text, ai_bgm_player, final_video_1, final_video_2, 
            app_state, start_button, results_group, result_title, 
            result_keywords, result_description
        ]
    )
    
    # [MODIFIED] Update the outputs list for finalize_button.click
    finalize_button.click(
        fn=finalize_manual_mode, 
        inputs=[app_state, custom_bgm_upload],
        outputs=[
            status_textbox, manual_mode_controls, final_video_1, final_video_2, 
            start_button, app_state, results_group, result_title,
            result_keywords, result_description
        ]
    )

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')
    demo.launch()