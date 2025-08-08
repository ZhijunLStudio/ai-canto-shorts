# config.py (V3.1 - Final Unified Version, Strictly Following User Values)
import os
from dotenv import load_dotenv

# ==============================================================================
# ---                      环境和顶级设置 (请先配置)                       ---
# ==============================================================================
load_dotenv()

# 【1. 源视频文件路径】
# 在Gradio UI模式下，此路径会被用户上传的文件覆盖。
# 在本地运行模式 (run_local.py) 下，此路径是必须指定的源视频。
VIDEO_PATH = "./testdata/demovideo.mp4"

# ==============================================================================
# ---                   核心认证信息 (必须替换为自己的)                     ---
# ==============================================================================

# 【2. 视觉分析大模型 (Vision LLM)】 - 用于第0步：分析视频帧
VIDEO_LLM_API_KEY = "your_video_llm_api_key"
VIDEO_LLM_BASE_URL = "your_video_llm_base_url"
VIDEO_LLM_MODEL_NAME = "o4-mini"

# 【3. Agent工作流大模型 (Text LLM)】 - 用于第1步及以后：生成剪辑计划、旁白等
AGENT_LLM_API_KEY = "your_agent_llm_api_key"
AGENT_LLM_BASE_URL = "your_agent_llm_base_url"
AGENT_LLM_MODEL_NAME = "ernie-4.5-turbo-128k-preview"

# 【4. 语音合成 (TTS)】 - 用于生成旁白
BAIDU_TTS_APP_ID = "your_baidu_tts_app_id"
BAIDU_TTS_API_KEY = "your_baidu_tts_api_key"
BAIDU_TTS_SECRET_KEY = "your_baidu_tts_secret_key"
TTS_OPTIONS = {'lan': 'zh', 'per': 20101} # 粤语

# ==============================================================================
# ---               阶段零: 视频/音频预分析参数                          ---
# ==============================================================================
VIDEO_ANALYSIS_CHUNK_S = 30           # 视觉分析时，每次送给LLM的视频块时长（秒）
VIDEO_ANALYSIS_CONCURRENCY = 4        # 视觉分析的并行请求数
AUDIO_ANALYSIS_CHUNK_MS = 10000       # 音频识别时，每次送给ASR的音频块时长（毫秒）
AUDIO_ANALYSIS_OVERLAP_MS = 1000      # 音频识别块之间的重叠时长
AUDIO_ANALYSIS_CONCURRENCY = 10       # 音频识别的并行请求数
SKIP_FIRST_N_SECONDS = 30             # 跳过视频开头的秒数，以规避片头

# ==============================================================================
# ---               阶段一: AI Agent 工作流参数 (编剧)                     ---
# ==============================================================================
# LLM 生成参数 (用于Agent工作流)
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.8
LLM_MAX_TOKENS = 4096
PENALTY_SCORE = 1.0

# Agent 视频内容查询参数
ANALYSIS_CHUNK_SECONDS = 300          # Agent在筛选片段时，每次查询的数据时间跨度
MAX_INDIVIDUAL_CLIP_DURATION_SECONDS = 20 # Agent筛选时，单个片段的最大时长

# 剪辑计划目标
TARGET_SHORT_VIDEO_DURATION_SECONDS = 140 # 最终成片的总目标时长（秒）

# ==============================================================================
# ---                   阶段二: 智能运镜参数 (导演&剪辑师)                 ---
# ==============================================================================

# 【1. 姿态识别模型】
MODEL_PATH = 'yolo11m-pose.pt'      # 本地YOLOv8/v11 Pose模型路径
MIN_KEYPOINT_CONFIDENCE = 0.5       # 关键点的最低置信度
MIN_VISIBLE_KEYPOINTS = 3           # 一个人至少有3个可见关键点才被认为是有效目标

# 【2. 镜头行为】
TARGET_ASPECT_RATIO = 9 / 16        # 目标视频宽高比 (竖屏)
CAMERA_SMOOTHING_FACTOR = 0.08      # 镜头平滑移动的平滑度。值越小，移动越平滑。
ADAPTIVE_KEN_BURNS_ENABLED = True   # 是否为无人物的场景启用缓慢的平移效果

# 【3. 主角选择】
MIN_SHOT_DURATION_SECONDS = 1.5     # 锁定一个主角后，至少拍摄1.5秒才允许切换
PROTAGONIST_SWITCH_THRESHOLD = 1.5  # 挑战者的活跃度需比当前主角高出50%，才能抢夺镜头
ACTIVITY_HISTORY_SIZE = 15          # 用于计算活跃度的历史帧数
# 关键点权重：手、脚等末端关节权重更高，以更好地捕捉“动作幅度”
KEYPOINT_WEIGHTS = { 5: 1.5, 6: 1.5, 7: 2.0, 8: 2.0, 9: 3.0, 10: 3.0 }
DEFAULT_WEIGHT = 1.0

# 【4. 对焦逻辑】
# 头部关键点索引 (COCO format)，用于“面部优先”对焦
HEAD_KEYPOINT_INDICES = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4
}

# ==============================================================================
# ---                        输出与调试 (文件管理)                         ---
# ==============================================================================
BASE_OUTPUT_DIR = "results"
CREATE_DEBUG_VIDEO = False  # 是否生成带有关节点和裁切框的调试视频
VIDEO_OUTPUT_STYLES = ['BLUR_COMPOSITE', 'CROP_VERTICAL'] # 输出视频的风格
FINAL_OUTPUT_RESOLUTION = (1080, 1920) # 最终视频分辨率
BLUR_KERNEL_SIZE = 41               # 模糊背景风格的模糊程度（必须是奇数）
SUBTITLE_LINE_LENGTH_VERTICAL = 12  # 字幕每行的最大字数

# 网络请求重试参数
MAX_RETRIES = 3
RETRY_DELAY_S = 5