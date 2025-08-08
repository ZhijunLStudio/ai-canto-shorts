# src/main_workflow.py (V7.1 - Scene Diversity Enhancement)

import json, logging, re, os
from typing import Dict, Iterator, List
from collections import Counter, defaultdict
from agno.models.openai import OpenAIChat
from agno.agent import Agent, RunResponse
from agno.workflow import Workflow
from datetime import datetime, timezone
import config
from src.tools import data_manager

# --- 辅助函数 (Unchanged) ---
def create_configured_llm(**kwargs) -> OpenAIChat:
    """创建配置好的LLM实例。"""
    config_dict = { "id": config.AGENT_LLM_MODEL_NAME, "api_key": config.AGENT_LLM_API_KEY, "base_url": config.AGENT_LLM_BASE_URL, "default_headers": {"Date": datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')} }
    config_dict.update(kwargs)
    return OpenAIChat(**config_dict)

def extract_json_from_response(text: str) -> dict or list:
    match = re.search(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', text, re.DOTALL)
    if match: json_str = match.group(1)
    else:
        match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
        if not match: return {} if not text.strip().startswith('[') else []
        json_str = match.group(0)
    try: return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"初步JSON解析失败: {e}。尝试修复并重试...")
        try:
            fixed_str = re.sub(r'(?<!\\)\n', r'\\n', json_str)
            fixed_str = re.sub(r',\s*([\}\]])', r'\1', fixed_str)
            return json.loads(fixed_str)
        except json.JSONDecodeError as e2:
            logging.error(f"即使在修复后，JSON解析仍然失败: {e2}"); return {}

def sanitize_clips(clips: List[Dict]) -> List[Dict]:
    if not isinstance(clips, list): logging.warning(f"sanitize_clips接收到的不是列表: {type(clips)}"); return []
    sanitized = []
    for clip in clips:
        if not isinstance(clip, dict): continue
        if 'start' in clip and 'start_time' not in clip: clip['start_time'] = clip['start']
        if 'end' in clip and 'end_time' not in clip: clip['end_time'] = clip['end']
        if 'description' in clip and 'reason' not in clip: clip['reason'] = clip['description']
        try:
            if 'start_time' in clip and 'end_time' in clip:
                clip['start_time'] = int(float(clip['start_time']))
                clip['end_time'] = int(float(clip['end_time']))
                sanitized.append(clip)
            else: logging.warning(f"片段缺少 'start_time' 或 'end_time' 键，已跳过: {clip}")
        except (ValueError, TypeError) as e: logging.warning(f"无法清洗片段，已跳过。错误: {e}, 片段: {clip}")
    return sanitized

def calculate_duration(clips: List[Dict]) -> int:
    return sum(c.get('end_time', 0) - c.get('start_time', 0) for c in clips)

def merge_adjacent_clips(clips: List[Dict], max_gap_seconds: int = 3) -> List[Dict]:
    if not clips: return []
    valid_clips = [c for c in clips if 'start_time' in c and 'end_time' in c]
    if not valid_clips: return []
    sorted_clips = sorted(valid_clips, key=lambda x: x.get('start_time', 0))
    merged = [sorted_clips[0]]
    for current_clip in sorted_clips[1:]:
        last_merged_clip = merged[-1]
        gap = current_clip.get('start_time', 0) - last_merged_clip.get('end_time', 0)
        if gap <= max_gap_seconds:
            last_merged_clip['end_time'] = current_clip.get('end_time', last_merged_clip.get('end_time'))
            if 'reason' in last_merged_clip and 'reason' in current_clip and last_merged_clip['reason'] and current_clip['reason']:
                last_merged_clip['reason'] += f" & {current_clip.get('reason', '')}"
        else: merged.append(current_clip)
    logging.info(f"片段合并完成：从 {len(clips)} 个片段合并为 {len(merged)} 个。")
    return merged

# [NEW] Helper function to group clips by scene keywords
def group_clips_by_scene(clips: List[Dict]) -> Dict[str, List[Dict]]:
    """Groups clips based on keywords in their 'reason' to promote scene diversity."""
    scene_groups = defaultdict(list)
    # Simple keywords that often indicate a scene type
    scene_keywords = ["天台", "夜街", "街道", "办公室", "室内", "车内", "黑板", "对话", "互动", "分享"]
    
    for clip in clips:
        reason = clip.get("reason", "")
        found_key = "其他场景"
        for key in scene_keywords:
            if key in reason:
                found_key = f"场景: {key}"
                break
        scene_groups[found_key].append(clip)
    
    logging.info(f"已将片段按场景聚类: {list(scene_groups.keys())}")
    return dict(scene_groups)


class VideoToShortsWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        logging.info("VideoToShortsWorkflow 已初始化 (V7.1 - Scene Diversity Enhancement)。")

    def run(self) -> Iterator[RunResponse]:
        model_call_params = {"temperature": 0.2, "top_p": 0.8, "max_tokens": 8000}
        
        logging.info("--- 工作流开始 ---")
        yield RunResponse(content="--- 工作流开始 ---\n")

        try:
            video_duration = data_manager.get_total_duration_seconds()
            
            # === 步骤 1 & 2: 筛选与审阅 (Unchanged) ===
            logging.info("--- 步骤 1: 分析师正在进行分块筛选... ---")
            yield RunResponse(content="--- 步骤 1: 分块筛选... ---\n")
            analyst_agent = Agent(model=create_configured_llm(**model_call_params), instructions=["你是一位顶级的视频分析师，任务是从给定的【视频内容块】中，找出所有具信息量和视觉吸引力的精华片段。", "每个片段的时长应在10到30秒之间。", "你的输出必须是一个JSON列表，每个对象必须严格包含 `start_time`, `end_time`, `reason` 三个键。", "如果内容块没有实质内容，就返回一个空列表 `[]`。"], debug_mode=True)
            candidate_clips = []
            for start_sec in range(0, video_duration, config.ANALYSIS_CHUNK_SECONDS):
                end_sec = min(start_sec + config.ANALYSIS_CHUNK_SECONDS, video_duration)
                content_chunk = data_manager.get_content_by_time(start_sec, end_sec)
                response = analyst_agent.run(f"这是从 {start_sec}s 到 {end_sec}s 的视频内容块。请严格按照指示，找出其中所有精华片段。\n\n【视频内容块】:\n{content_chunk}")
                found_clips = sanitize_clips(extract_json_from_response(response.content))
                if found_clips: candidate_clips.extend(found_clips)
            self.session_state["candidate_clips"] = sorted(candidate_clips, key=lambda x: x.get('start_time', 0))
            logging.info(f"✅ 步骤 1 完成. 初步找到 {len(candidate_clips)} 个候选片段。")
            yield RunResponse(content="✅ 初步筛选完成。\n")

            logging.info("--- 步骤 2: 审阅Agent正在过滤无效片段... ---")
            yield RunResponse(content="--- 步骤 2: 过滤无效片段... ---\n")
            reviewer_agent = Agent(model=create_configured_llm(temperature=0.1), instructions=["你是一位严格的内容审阅员，任务是审查视频片段列表并移除所有无效内容，如片头曲、片尾、纯粹的过渡画面、广告等。", "你的输出必须是**只包含有效片段**的JSON列表。"], debug_mode=True)
            response = reviewer_agent.run(f"这是初步筛选的视频片段列表。请审查并移除所有无效内容，只保留有效的正片内容。\n\n待审阅的片段列表:\n```json\n{json.dumps(self.session_state['candidate_clips'], indent=2, ensure_ascii=False)}\n```\n\n请直接返回过滤后的有效片段JSON列表。")
            reviewed_clips = sanitize_clips(extract_json_from_response(response.content))
            self.session_state["reviewed_clips"] = reviewed_clips
            logging.info(f"✅ 步骤 2 完成. 从 {len(candidate_clips)} 个片段中过滤出 {len(reviewed_clips)} 个有效片段。")
            yield RunResponse(content="✅ 无效片段过滤完成。\n")

            # === 步骤 3: 故事规划 (Director Agent) - [MODIFIED with Scene Diversity] ===
            logging.info("--- 步骤 3: 总导演正在规划故事线... ---")
            yield RunResponse(content="--- 步骤 3: 规划故事线... ---\n")
            director_agent = Agent(
                model=create_configured_llm(temperature=0.4),
                instructions=[
                    "你是一位经验丰富的总导演。你的任务是从一系列【按场景分组的候选片段】中，挑选并组合出一个结构完整、叙事流畅的短视频。",
                    "**[核心规则]**:",
                    "1. **场景多样性优先**: 你的首要任务是确保最终成片**尽可能包含来自不同场景的片段**。避免选择太多外观相似的场景。",
                    "2. **故事结构**: 你的选择应兼顾视频的开端、发展和结尾，讲述一个连贯的故事。",
                    "3. **片段数量**: 最终成片必须包含 **5 到 7 个**不同的片段。",
                    "4. **目标时长**: 最终总时长必须尽可能精确地等于 **{} 秒**。".format(config.TARGET_SHORT_VIDEO_DURATION_SECONDS),
                    "**输出格式要求**: 你的输出必须是一个JSON列表，每个对象都必须包含 `start_time`, `end_time`, 和 `reason` 这三个键。"
                ],
                debug_mode=True
            )
            
            valid_clips_for_director = self.session_state["reviewed_clips"]
            
            # [NEW] Group clips by scene before sending to the agent
            grouped_scenes = group_clips_by_scene(valid_clips_for_director)
            
            # Create a formatted string for the prompt
            scene_prompt_string = ""
            for scene, clips_in_scene in grouped_scenes.items():
                scene_prompt_string += f"### {scene}\n"
                scene_prompt_string += f"```json\n{json.dumps(clips_in_scene, indent=2, ensure_ascii=False)}\n```\n\n"

            director_prompt = (
                f"视频总时长为 {video_duration} 秒。我已将所有有效片段按场景分组。请严格遵守你的核心规则，特别是【场景多样性优先】，挑选5到7个来自不同场景的片段，组合成一个总时长约 {config.TARGET_SHORT_VIDEO_DURATION_SECONDS} 秒的精彩短片。\n\n"
                f"【按场景分组的候选片段】:\n{scene_prompt_string}"
                "请直接返回你最终选择的片段JSON列表。"
            )

            response = director_agent.run(director_prompt)
            final_planned_clips = sanitize_clips(extract_json_from_response(response.content))
            self.session_state["final_clips"] = merge_adjacent_clips(final_planned_clips)
            logging.info(f"✅ 步骤 3 完成. 规划了 {calculate_duration(self.session_state['final_clips'])}秒 的视频，包含 {len(self.session_state['final_clips'])} 个片段。")
            yield RunResponse(content="✅ 故事线规划完成。\n")

            # === 步骤 4, 5, 6, 7 (Unchanged from V7.0) ===
            logging.info("--- 步骤 4: 确定主角... ---")
            yield RunResponse(content="--- 步骤 4: 确定主角... ---\n")
            selected_clips_content = ""
            for clip in self.session_state["final_clips"]:
                selected_clips_content += f"\n--- 片段 ({clip['start_time']}s - {clip['end_time']}s) ---\n"
                selected_clips_content += data_manager.get_content_by_time(clip['start_time'], clip['end_time'])
            
            casting_agent = Agent(model=create_configured_llm(temperature=0.1), instructions=["你是一位选角导演，请根据提供的【已选定片段内容】，找出其中最核心、出现最频繁的主角。", "你的任务是给这个主角一个简洁的描述。"], debug_mode=True)
            response = casting_agent.run(f"这是最终选定的视频片段内容。请确定故事的主角是谁。\n\n【已选定片段内容】:\n{selected_clips_content}\n\n请只返回主角的描述。")
            main_narrator_desc = response.content.strip()
            self.session_state["main_narrator_description"] = main_narrator_desc
            logging.info(f"✅ 步骤 4 完成. 确定主角为: {main_narrator_desc}")
            yield RunResponse(content="✅ 主角确定完成。\n")

            logging.info("--- 步骤 5: 逐片段创作旁白... ---")
            yield RunResponse(content="--- 步骤 5: 逐片段创作旁白... ---\n")
            narrator_agent = Agent(
                model=create_configured_llm(temperature=0.75, max_tokens=500),
                instructions=[
                    "你是一位顶级的粤语剧本作家，任务是**扮演**指定的【主要叙事角色】。",
                    "**[核心要点]**:",
                    "1. **角色一致性**: 你的所有旁白都必须严格维持【你扮演的角色】的身份、口吻和心路历程。",
                    "2. **内部视角**: 你要描述的是角色“我”的所见、所闻、所感、所想，而不是作为一个局外人在解说画面。",
                    "3. **避免套路**: 严禁在不同的旁白中使用相似的句式或开头。",
                    "4. **地道粤语**: 输出必须是地道、流畅的粤语口语。",
                ],
                debug_mode=True
            )
            clips_with_narration = []
            for i, clip in enumerate(self.session_state["final_clips"]):
                clip_content = data_manager.get_content_by_time(clip.get('start_time'), clip.get('end_time'))
                previous_narrations = [c.get('narration', '') for c in clips_with_narration if c.get('narration')]
                if not previous_narrations:
                    previous_narrations_context = "这是第一个场景，故事从这里开始。"
                else:
                    narration_lines = [f"{idx+1}. \"{narration}\"" for idx, narration in enumerate(previous_narrations)]
                    previous_narrations_context = ("为了确保故事连贯，以下是到目前为止已经创作好的旁白，请你接续下去：\n" + "\n".join(narration_lines))

                prompt = (
                    f"你要扮演的角色是：【{main_narrator_desc}】。\n\n"
                    f"**上下文回顾**: \n{previous_narrations_context}\n\n"
                    f"**当前场景({clip.get('start_time')}s-{clip.get('end_time')}s)内容**: \n{clip_content}\n\n"
                    "--- **创作要求** ---\n"
                    "1. 以第一人称“我”的视角，为**当前场景**创作一段能**流畅衔接上下文**的旁白。\n"
                    "2. **语言要求**: 必须是**地道的粤语口语**。\n"
                    "3. **输出要求**: 只输出纯净的旁白文本，不要任何额外解释或引号。\n"
                    "4. **避免重复**: 严禁在多个片段的旁白中使用雷同的开头或句式。"
                )
                response = narrator_agent.run(prompt)
                narration_text = response.content.strip().strip('"')
                clip_with_narration = clip.copy(); clip_with_narration["narration"] = narration_text
                clips_with_narration.append(clip_with_narration)
            
            self.session_state["final_output_clips"] = clips_with_narration
            logging.info(f"✅ 步骤 5 完成. 已为所有 {len(clips_with_narration)} 个片段生成独立旁白。")
            yield RunResponse(content="✅ 逐片段旁白创作完成。\n")

            logging.info("--- 步骤 6: 创作营销内容... ---")
            yield RunResponse(content="--- 步骤 6: 创作营销内容... ---\n")
            marketing_agent = Agent(model=create_configured_llm(temperature=0.7), instructions=["你是顶级的社交媒体推广专员，擅长用生动的粤语为短视频创作吸引人的标题、标签和简介。","你的任务是根据剪辑方案，创作标题(title), 关键词(keywords), 和简介(description)。","你的输出必须是 `{\"title\": \"...\", \"keywords\": [\"...\"], \"description\": \"...\"}` 格式的JSON对象。"], debug_mode=True)
            response = marketing_agent.run(f"这是最终的短视频剪辑方案，请为此创作标题、关键词和简介。\n**风格要求**:\n - **标题**: **严格由两句短句构成的精华标题**。总字数控制在20字以内。\n - **关键词**: 3-5个精准的标签。\n - **简介**: 口语化、亲切地概括视频亮点，并用`#`带上热门hashtag。\n\n**【短视频剪辑方案】**:\n{json.dumps(self.session_state['final_output_clips'], indent=2, ensure_ascii=False)}\n\n请严格按照JSON格式输出。")
            marketing_content = extract_json_from_response(response.content)
            if not isinstance(marketing_content, dict) or not all(k in marketing_content for k in ["title", "keywords", "description"]):
                raise ValueError("营销内容创作步骤未能生成有效结果。")
            self.session_state["marketing_content"] = marketing_content
            logging.info("✅ 步骤 6 完成. 营销内容已生成。")
            yield RunResponse(content="✅ 营销内容创作完成。\n")

            logging.info("--- 步骤 7: 生成BGM关键词... ---")
            yield RunResponse(content="--- 步骤 7: 生成BGM关键词... ---\n")
            bgm_keyword_agent = Agent(model=create_configured_llm(temperature=0.3), instructions=["你是专业的音乐总监，任务是为短视频构思最合适的BGM搜索词。","你的输出**必须**是 `{\"bgm_keywords\": \"...\"}` 格式的JSON对象。","关键词**必须**是一个**单一的、不含逗号的英文搜索短语**。"], debug_mode=True)
            final_clips_summary = "\n".join([f"- {c.get('narration', '')}" for c in self.session_state.get("final_output_clips", [])])
            response = bgm_keyword_agent.run(f"这是短视频的核心内容，请为此构思最合适的BGM搜索关键词。\n\n**核心内容/旁白摘要**:\n{final_clips_summary}\n\n请严格按格式输出。")
            bgm_keywords_data = extract_json_from_response(response.content)
            if isinstance(bgm_keywords_data, dict) and "bgm_keywords" in bgm_keywords_data:
                self.session_state["bgm_keywords"] = bgm_keywords_data["bgm_keywords"]
                logging.info(f"✅ 步骤 7 完成. BGM关键词: {self.session_state['bgm_keywords']}")
            else:
                self.session_state["bgm_keywords"] = None; logging.warning("BGM关键词生成失败。")
            yield RunResponse(content="✅ BGM关键词生成完成。\n")

            logging.info("--- 最终合成... ---")
            final_output = { **self.session_state.get('marketing_content', {}), "bgm_keywords": self.session_state.get("bgm_keywords"), "clips": self.session_state.get("final_output_clips", []) }
            self.session_state["final_output"] = final_output
            logging.info("--- 工作流成功结束 ---")
            yield RunResponse(content=json.dumps(final_output, indent=2, ensure_ascii=False))

        except Exception as e:
            logging.critical("工作流在执行过程中捕获到严重错误。", exc_info=True)
            yield RunResponse(content=f"--- 工作流因错误而终止: {e} ---\n")
            self.session_state["final_output"] = None