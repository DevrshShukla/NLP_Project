import re
import unicodedata
from typing import List, Tuple

class TextCleaner:
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.IGNORECASE)
    PAGE_MARKER_PATTERN = re.compile(r'^(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+)$', re.IGNORECASE | re.MULTILINE)
    DOT_LEADER_PATTERN = re.compile(r'\.{4,}')
    HYPHENATED_LINEBREAK_PATTERN = re.compile(r'([A-Za-z]+)-\s*\n\s*([A-Za-z]+)')
    
    @classmethod
    def clean(cls, raw_text: str) -> str:
        if not raw_text:
            return ""
        
        # 1. Unicode NFKC normalization
        text = unicodedata.normalize("NFKC", raw_text)
        
        # Replace non-breaking spaces, curly quotes, dashes, ellipsis
        text = text.replace('\xa0', ' ')
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        text = text.replace('—', ' - ').replace('–', ' - ').replace('…', '...')
        
        # Remove null and non-printable control characters, but keep \n, \t, \r
        text = "".join(ch for ch in text if ch in ('\n', '\t', '\r') or (unicodedata.category(ch)[0] != 'C'))
        
        # 2. Noise removal: URLs, Emails, Dot leaders
        text = cls.URL_PATTERN.sub('', text)
        text = cls.EMAIL_PATTERN.sub('', text)
        text = cls.DOT_LEADER_PATTERN.sub(' ', text)
        
        # 3. Join hyphenated line breaks: classi-\nfication -> classification
        text = cls.HYPHENATED_LINEBREAK_PATTERN.sub(r'\1\2', text)
        
        # Clean line by line for page markers and isolated numbers
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                cleaned_lines.append("")
                continue
            if cls.PAGE_MARKER_PATTERN.match(line_str):
                continue
            if line_str.isdigit() and len(line_str) <= 4:
                continue
            cleaned_lines.append(line)
            
        text = "\n".join(cleaned_lines)
        
        # 4. Paragraph and newline formatting:
        # Convert single newlines to spaces, keep double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        clean_paragraphs = []
        for p in paragraphs:
            # Replace single newlines with space and collapse whitespace
            single_line = re.sub(r'\s*\n\s*', ' ', p).strip()
            single_line = re.sub(r'[ \t]+', ' ', single_line)
            if single_line:
                clean_paragraphs.append(single_line)
                
        final_text = "\n\n".join(clean_paragraphs)
        return final_text
    
    @classmethod
    def deduplicate_sentences(cls, sentences: List[str]) -> List[Tuple[str, int]]:
        """
        Returns list of (sentence, original_index) retaining only the first occurrence 
        of exact duplicate sentences (case-insensitive).
        """
        seen = set()
        deduped = []
        for idx, s in enumerate(sentences):
            normalized = s.strip().lower()
            if not normalized:
                continue
            if normalized not in seen:
                seen.add(normalized)
                deduped.append((s, idx))
        return deduped
