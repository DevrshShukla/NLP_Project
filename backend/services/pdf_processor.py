import fitz  # PyMuPDF
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter
from backend.core.errors import create_app_error

class PDFProcessor:
    @classmethod
    def process_pdf_bytes(cls, pdf_bytes: bytes, max_mb: int = 10, max_pages: int = 200) -> Dict[str, Any]:
        # 1. File size check
        size_mb = len(pdf_bytes) / (1024 * 1024)
        if size_mb > max_mb:
            raise create_app_error("FILE_TOO_LARGE")

        # 2. Magic bytes validation
        if not pdf_bytes.startswith(b'%PDF-'):
            raise create_app_error("INVALID_FILE_TYPE")

        # 3. Open in memory via PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception:
            raise create_app_error("INVALID_PDF")

        # 4. Check password protection
        if doc.is_encrypted or doc.needs_pass:
            raise create_app_error("PASSWORD_PROTECTED_PDF")

        # 5. Check page limits
        total_pages = doc.page_count
        if total_pages > max_pages:
            raise create_app_error("DOCUMENT_TOO_LARGE")
        if total_pages == 0:
            raise create_app_error("NO_READABLE_TEXT")

        # 6. Extract text per page and compute quality
        page_texts: List[str] = []
        readable_pages_count = 0
        all_header_lines: List[str] = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text", sort=True) or ""
            page_texts.append(text)

            # Collect candidate header/footer lines (first 2 and last 2 lines)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                all_header_lines.append(lines[0])
                if len(lines) > 1:
                    all_header_lines.append(lines[-1])

            # Check if page is readable: at least 50 characters and >= 60% alphabetic
            clean_page_chars = [c for c in text if not c.isspace()]
            if len(clean_page_chars) >= 50:
                alpha_count = sum(1 for c in clean_page_chars if c.isalpha())
                if (alpha_count / len(clean_page_chars)) >= 0.60:
                    readable_pages_count += 1

        quality = round(readable_pages_count / total_pages, 2)
        warnings: List[str] = []

        # Find repeated header/footer lines (> 60% of pages)
        header_counter = Counter(all_header_lines)
        repeated_headers = {
            line for line, count in header_counter.items()
            if count > (0.60 * total_pages) and len(line) < 100
        }

        # Clean pages of repeated headers and join
        cleaned_page_texts = []
        for text in page_texts:
            lines = text.splitlines()
            filtered_lines = [l for l in lines if l.strip() not in repeated_headers]
            cleaned_page_texts.append("\n".join(filtered_lines))

        full_extracted_text = "\n\n".join(cleaned_page_texts).strip()

        # Check total extracted text
        if len(full_extracted_text) < 50 or quality == 0:
            raise create_app_error("NO_READABLE_TEXT")

        if quality < 0.5:
            warnings.append("LOW_EXTRACTION_QUALITY")

        return {
            "text": full_extracted_text,
            "page_count": total_pages,
            "quality": quality,
            "char_count": len(full_extracted_text),
            "word_count": len(full_extracted_text.split()),
            "warnings": warnings
        }
