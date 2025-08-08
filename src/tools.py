# src/tools.py (V2.0 - Refactored for global data_manager)

import logging

# [FIXED] Import the GLOBAL INSTANCE from data_manager, do not initialize here.
from src.data_manager import data_manager

logger = logging.getLogger(__name__)

# --- NO INITIALIZATION BLOCK NEEDED ANYMORE ---

def get_video_summary_and_duration() -> str:
    """
    获取整个视频的剧情摘要和总时长（秒）。Agent 在开始规划时应首先调用此工具。
    """
    if not data_manager.is_initialized:
        return "错误：数据管理器未被主程序正确初始化。"
    
    logger.info("--- 🛠️ 调用工具: get_video_summary_and_duration ---")
    summary = data_manager.get_full_summary()
    duration = data_manager.get_total_duration_seconds()
    observation_content = f"视频总时长: {duration}秒。\n剧情摘要:\n{summary}"
    logger.debug("--- 📄 将要返回给 Agent 的 Observation 内容 ---\n%s", observation_content)
    return observation_content


def query_detailed_content(start_seconds: int, end_seconds: int) -> str:
    """
    获取指定时间范围（秒）内的详细数据，包括视觉描述和对白。
    """
    if not data_manager.is_initialized:
        return "错误：数据管理器未被主程序正确初始化。"

    logger.info(f"--- 🛠️ 调用工具: query_detailed_content, start={start_seconds}, end={end_seconds} ---")
    
    if start_seconds >= end_seconds:
        return f"错误：开始时间 {start_seconds} 必须小于结束时间 {end_seconds}。"
    
    total_duration = data_manager.get_total_duration_seconds()
    if start_seconds > total_duration:
        return f"错误：开始时间 {start_seconds} 超过视频总时长 {total_duration} 秒。"
    
    content = data_manager.get_content_by_time(start_seconds, end_seconds)
    logger.debug("--- 📄 将要返回给 Agent 的 Observation 内容 (前1000字符): ---\n%s",
                 content[:1000] + "..." if len(content) > 1000 else content)
    return content