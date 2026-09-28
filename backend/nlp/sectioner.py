import re
from typing import List, Dict, Any, Optional

class Section:
    def __init__(self, index: int, heading: Optional[str], text: str, start_char: int, end_char: int):
        self.index = index
        self.heading = heading
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.token_count = len(text.split())

class Sectioner:
    NUMBERED_HEADING_PATTERN = re.compile(
        r'^(chapter\s+\d+|unit\s+[ivxlcdm\d]+|\d+(\.\d+)*)\s*[:\.\-]?\s*',
        re.IGNORECASE
    )
    
    @classmethod
    def is_heading(cls, line: str) -> bool:
        line_clean = line.strip()
        if not line_clean:
            return False
        
        words = line_clean.split()
        if len(words) > 12 or len(words) == 0:
            return False
        
        # Must not end with a terminal period
        if line_clean.endswith('.'):
            return False
        
        # Starts with numbered section pattern
        if cls.NUMBERED_HEADING_PATTERN.match(line_clean):
            return True
            
        # Is ALL CAPS (at least 3 letters)
        letters = [c for c in line_clean if c.isalpha()]
        if len(letters) >= 3 and line_clean == line_clean.upper():
            return True
            
        # Is Title Case (at least 60% of words start with capital letter)
        cap_words = [w for w in words if w[0].isupper()]
        if len(words) >= 1 and (len(cap_words) / len(words)) >= 0.6:
            return True
            
        return False

    @classmethod
    def split_into_sections(cls, text: str) -> List[Section]:
        paragraphs = text.split("\n\n")
        raw_sections = []
        current_heading = None
        current_body = []
        current_start = 0
        char_cursor = 0
        
        for p in paragraphs:
            p_len = len(p)
            lines = p.strip().split("\n")
            if len(lines) == 1 and cls.is_heading(lines[0]):
                if current_body:
                    sec_text = "\n\n".join(current_body)
                    raw_sections.append({
                        "heading": current_heading,
                        "text": sec_text,
                        "start": current_start,
                        "end": current_start + len(sec_text)
                    })
                    current_body = []
                current_heading = lines[0].strip()
                current_start = char_cursor
            else:
                current_body.append(p)
            char_cursor += p_len + 2  # accounted for \n\n
            
        if current_body:
            sec_text = "\n\n".join(current_body)
            raw_sections.append({
                "heading": current_heading,
                "text": sec_text,
                "start": current_start,
                "end": current_start + len(sec_text)
            })
            
        # If no headings found, or only 1 section with long text, window it
        if not raw_sections:
            raw_sections = [{
                "heading": None,
                "text": text,
                "start": 0,
                "end": len(text)
            }]
            
        # Merge sections shorter than 80 tokens with previous section if possible
        merged_sections = []
        for sec in raw_sections:
            token_count = len(sec["text"].split())
            if merged_sections and token_count < 80:
                merged_sections[-1]["text"] += "\n\n" + sec["text"]
                merged_sections[-1]["end"] = sec["end"]
            else:
                merged_sections.append(sec)
                
        # Split sections longer than 1200 tokens (Section 15: windows of ~800 tokens with 100 token overlap)
        final_sections: List[Section] = []
        sec_idx = 0
        for sec in merged_sections:
            tokens = sec["text"].split()
            if len(tokens) <= 1200:
                final_sections.append(Section(
                    index=sec_idx,
                    heading=sec["heading"],
                    text=sec["text"],
                    start_char=sec["start"],
                    end_char=sec["end"]
                ))
                sec_idx += 1
            else:
                # Windowing
                window_size = 800
                overlap = 100
                step = window_size - overlap
                start_tok = 0
                while start_tok < len(tokens):
                    end_tok = min(start_tok + window_size, len(tokens))
                    chunk_text = " ".join(tokens[start_tok:end_tok])
                    final_sections.append(Section(
                        index=sec_idx,
                        heading=sec["heading"],
                        text=chunk_text,
                        start_char=sec["start"],  # approximate
                        end_char=sec["end"]
                    ))
                    sec_idx += 1
                    if end_tok == len(tokens):
                        break
                    start_tok += step

        return final_sections
