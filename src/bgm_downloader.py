import os
import logging
from music_dl import config, source

# 配置日志记录，可以根据需要调整 music_dl 的日志级别，避免过多输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s')
logging.getLogger("music_dl").setLevel(logging.WARNING)

def download_first_successful_song(keyword: str, outdir: str, try_count: int = 15) -> str | None:
    """
    根据关键词搜索歌曲，并下载第一首能够成功下载的歌曲。
    会按顺序尝试下载搜索结果中的前 try_count 首歌曲。

    Args:
        keyword (str): 音乐搜索关键词。
        outdir (str): 下载文件的输出目录。
        try_count (int): 最大尝试下载的歌曲数量。

    Returns:
        str | None: 成功下载的歌曲文件路径，如果所有尝试都失败则返回 None。
    """
    logger = logging.getLogger("BGMDownloader")
    logger.info(f"开始BGM搜索，关键词: '{keyword}'")

    # 1. 初始化 music_dl 配置
    try:
        config.init()
    except Exception as e:
        logger.error(f"初始化 music_dl 配置失败: {e}")
        return None
        
    # 2. 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 3. 设置 music_dl 参数
    config.set("keyword", keyword)
    config.set("outdir", outdir)
    config.set("lyrics", False)  # 我们不需要歌词
    config.set("cover", False)   # 我们不需要封面
    config.set("number", try_count) # 需要获取的搜索结果数量

    # 4. 从稳定源搜索歌曲
    stable_sources = ["netease", "kugou", "baidu", "qq", "migu"]
    music_source = source.MusicSource()
    logger.info(f"搜索源: {', '.join(s.upper() for s in stable_sources)}")
    
    try:
        song_list = music_source.search(keyword, stable_sources)
    except Exception as e:
        logger.error(f"音乐搜索过程中发生异常: {e}")
        return None

    if not song_list:
        logger.warning(f"未找到与关键词 '{keyword}' 相关的歌曲。")
        return None

    logger.info(f"找到 {len(song_list)} 首潜在歌曲。将逐一尝试下载...")

    # 5. 遍历并尝试下载
    songs_to_try = song_list[:try_count]
    for i, song in enumerate(songs_to_try, 1):
        logger.info(f"--- [尝试下载 {i}/{len(songs_to_try)}] '{song.title} - {song.singer}' ---")
        try:
            # 记录下载前的目录状态，以便找到新文件
            files_before = set(os.listdir(outdir))
            
            # 调用核心下载方法
            song.download()

            files_after = set(os.listdir(outdir))
            new_files = files_after - files_before

            if not new_files:
                 logger.warning(f"  -> 下载调用完成，但未创建新文件。")
                 continue

            downloaded_filename = new_files.pop()
            downloaded_filepath = os.path.join(outdir, downloaded_filename)

            # 检查文件是否有效（例如，大于1KB）
            if os.path.exists(downloaded_filepath) and os.path.getsize(downloaded_filepath) > 1024:
                logger.info(f"  -> ✅ 下载成功！歌曲保存至: {downloaded_filepath}")
                return downloaded_filepath
            else:
                logger.warning(f"  -> ‼️ 下载的文件 '{downloaded_filename}' 无效（空或太小）。正在删除并尝试下一首。")
                if os.path.exists(downloaded_filepath):
                    os.remove(downloaded_filepath)
                continue

        except Exception as e:
            logger.error(f"  -> ❌ 下载 '{song.title}' 失败。原因: {e}")
            # 清理可能产生的失败文件
            files_after = set(os.listdir(outdir))
            new_files = files_after - files_before
            for f in new_files:
                os.remove(os.path.join(outdir, f))

    logger.error("所有下载尝试均失败，未能获取BGM。")
    return None