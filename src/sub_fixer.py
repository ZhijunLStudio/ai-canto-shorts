# src/sub_fixer.py (V1.1 - With Quote Removal)

import logging
import re
import srt

class SubtitleFixer:
    def __init__(self, line_length=15, remove_punctuation=False):
        self.line_length = line_length
        self.remove_punctuation = remove_punctuation
        self.logger = logging.getLogger(self.__class__.__name__)

    def _intelligent_wrap(self, text: str) -> str:
        lines = []
        current_line = ""
        # [MODIFIED] Also remove quotes before wrapping
        text_to_wrap = text.replace('『', '').replace('』', '')
        
        for part in text_to_wrap.split('\n'):
            words = list(part)
            for word in words:
                if len(current_line) + len(word) > self.line_length:
                    break_point = -1
                    for i, char in reversed(list(enumerate(current_line))):
                        if char in '，。！？、':
                            break_point = i + 1
                            break
                    if break_point != -1:
                        lines.append(current_line[:break_point])
                        current_line = current_line[break_point:]
                    else:
                        lines.append(current_line)
                        current_line = ""
                current_line += word
            if current_line:
                lines.append(current_line)
                current_line = ""
        return '\n'.join(line.strip() for line in lines if line.strip())

    def process_srt_file(self, input_srt_path: str, output_srt_path: str) -> bool:
        self.logger.info(f"Processing SRT file: {input_srt_path} (line_length={self.line_length})")
        try:
            with open(input_srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            subtitles = list(srt.parse(content))
            new_subtitles = []
            
            for sub in subtitles:
                # The wrapping function now handles quote removal
                wrapped_content = self._intelligent_wrap(sub.content)
                
                if self.remove_punctuation:
                    lines = wrapped_content.split('\n')
                    cleaned_lines = [re.sub(r'[，。！？、\s]+$', '', line) for line in lines]
                    wrapped_content = '\n'.join(cleaned_lines)
                    
                new_sub = srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=wrapped_content)
                new_subtitles.append(new_sub)
                
            final_srt_content = srt.compose(new_subtitles)
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(final_srt_content)
            self.logger.info(f"Subtitle processing complete -> {output_srt_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error processing SRT file {input_srt_path}: {e}", exc_info=True)
            return False