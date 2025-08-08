#!/bin/bash

set -e

# 1. 参数
INPUT_VIDEO="$1"
INPUT_SRT="$2"
FINAL_OUTPUT="$3"
# [MODIFIED] Reverted to use the pre-formatted title text directly
TITLE_TEXT="$4"
FONT_FILE="$5"
TEMP_DIR=$(dirname "$FINAL_OUTPUT")/temp_burn
mkdir -p "$TEMP_DIR"

# [REMOVED] Title sanitization is now handled by run_local.py

# 2. 依赖检查
if [ -z "$INPUT_VIDEO" ] || [ ! -f "$INPUT_VIDEO" ]; then echo "❌ 错误: 输入视频为空或不存在！"; exit 1; fi
if [ -z "$FONT_FILE" ] || [ ! -f "$FONT_FILE" ]; then echo "❌ 错误: 字体文件为空或不存在！"; exit 1; fi
if ! command -v ffmpeg &> /dev/null; then echo "❌ 错误: 找不到 ffmpeg 命令。"; exit 1; fi

# 3. ASS 样式 (No changes)
TITLE_FONT_SIZE=100
TITLE_PRIMARY_COLOR="&H00FFFFFF"
TITLE_OUTLINE_COLOR="&H00000000"
TITLE_OUTLINE_WIDTH=3.5
TITLE_TOP_MARGIN=250
SUB_FONT_SIZE=65
SUB_PRIMARY_COLOR="&H0000FFFF"
SUB_OUTLINE_COLOR="&H00000000"
SUB_OUTLINE_WIDTH=3.0
SUB_SHADOW_DEPTH=1.5
SUB_BOTTOM_MARGIN=150

# 4. 环境准备 (No changes)
SAFE_FONT_NAME="safe_font.ttf"
cp "$FONT_FILE" "$TEMP_DIR/$SAFE_FONT_NAME"
trap 'rm -rf "$TEMP_DIR"' EXIT
VIDEO_DIMS=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$INPUT_VIDEO")
VIDEO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO")
VIDEO_WIDTH=${VIDEO_DIMS%x*}
VIDEO_HEIGHT=${VIDEO_DIMS#*x}
END_TIME=$(awk -v dur="$VIDEO_DURATION" 'BEGIN{secs=int(dur); H=int(secs/3600); M=int((secs%3600)/60); S=secs%60; MS=int((dur-secs)*100); printf "%d:%02d:%02d.%02d", H, M, S, MS}')

# 5. 生成 ASS (No changes)
FILTER_COMPLEX_STRING="[0:v]"
if [ -n "$TITLE_TEXT" ]; then
    cat > "$TEMP_DIR/title.ass" << EOL
[Script Info]
PlayResX: ${VIDEO_WIDTH}
PlayResY: ${VIDEO_HEIGHT}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TitleStyle,${SAFE_FONT_NAME},${TITLE_FONT_SIZE},${TITLE_PRIMARY_COLOR},&H00FFFFFF,${TITLE_OUTLINE_COLOR},&H00000000,-1,0,0,0,100,100,0,0,1,${TITLE_OUTLINE_WIDTH},1,8,0,0,${TITLE_TOP_MARGIN},1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,${END_TIME},TitleStyle,,0,0,0,,${TITLE_TEXT}
EOL
    FILTER_COMPLEX_STRING="${FILTER_COMPLEX_STRING}subtitles=$TEMP_DIR/title.ass:fontsdir=$TEMP_DIR[v1]; [v1]"
fi
if [ -n "$INPUT_SRT" ] && [ -f "$INPUT_SRT" ]; then
    ffmpeg -nostdin -i "$INPUT_SRT" "$TEMP_DIR/subs_raw.ass" -y > /dev/null 2>&1
    (
    echo "[Script Info]";
    echo "PlayResX: ${VIDEO_WIDTH}"; echo "PlayResY: ${VIDEO_HEIGHT}";
    echo "[V4+ Styles]";
    echo "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding";
    echo "Style: TVBStyle,${SAFE_FONT_NAME},${SUB_FONT_SIZE},${SUB_PRIMARY_COLOR},&H00FFFFFF,${SUB_OUTLINE_COLOR},&H00000000,-1,0,0,0,100,100,0,0,1,${SUB_OUTLINE_WIDTH},${SUB_SHADOW_DEPTH},2,0,0,${SUB_BOTTOM_MARGIN},1";
    echo "[Events]";
    echo "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text";
    grep "Dialogue:" "$TEMP_DIR/subs_raw.ass" | sed -E 's/^(Dialogue: [^,]*,[^,]*,[^,]*),[^,]*(,.*)/\1,TVBStyle\2/';
    ) > "$TEMP_DIR/subs.ass"
    FILTER_COMPLEX_STRING="${FILTER_COMPLEX_STRING}subtitles=$TEMP_DIR/subs.ass:fontsdir=$TEMP_DIR"
else
    FILTER_COMPLEX_STRING="${FILTER_COMPLEX_STRING}null"
fi

# 6. 烧录 (No changes)
ffmpeg -nostdin -y -i "$INPUT_VIDEO" \
  -filter_complex "${FILTER_COMPLEX_STRING}[v_out]" \
  -map "[v_out]" -map 0:a? \
  -c:v libx264 -preset veryfast -crf 20 -c:a copy -pix_fmt yuv420p "$FINAL_OUTPUT"

echo "✅ 字幕烧录完成！"