import json
import math
import os
from typing import Dict, List, Set
from backend.nlp.ngram_extractor import Candidate

class TfidfService:
    def __init__(self, bg_idf_path: str):
        self.bg_terms: Dict[str, float] = {}
        self.max_bg_idf: float = 12.5129
        self.corpus_size: int = 100000
        
        if os.path.exists(bg_idf_path):
            try:
                with open(bg_idf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.bg_terms = data.get("terms", {})
                    self.max_bg_idf = data.get("max_idf", 12.5129)
                    self.corpus_size = data.get("corpus_size", 100000)
            except Exception as e:
                print(f"Warning: Failed to load background IDF from {bg_idf_path}: {e}")

    def get_bg_idf(self, term: str) -> float:
        # If multi-word, average IDF of individual tokens
        words = term.split()
        if len(words) == 1:
            return self.bg_terms.get(term, self.max_bg_idf)
        idfs = [self.bg_terms.get(w, self.max_bg_idf) for w in words]
        return sum(idfs) / len(idfs)

    def compute_tfidf(
        self,
        candidates: List[Candidate],
        total_sections: int
    ) -> Dict[str, float]:
        if not candidates:
            return {}
            
        N = max(1, total_sections)
        raw_tfidf: Dict[str, float] = {}
        
        for c in candidates:
            # Sublinear TF: 1 + ln(count)
            tf = 1.0 + math.log(c.count) if c.count > 0 else 0.0
            
            # Document IDF
            df = len(c.section_ids)
            idf_doc = math.log((1.0 + N) / (1.0 + df)) + 1.0
            
            # Background IDF
            idf_bg = self.get_bg_idf(c.key)
            
            # Blended IDF
            if N < 5:
                idf = idf_bg
            else:
                idf = 0.5 * idf_doc + 0.5 * idf_bg
                
            # Length boost: 1.00 for unigrams, 1.15 for bigrams, 1.25 for trigrams
            boost = c.length_boost
            
            raw_tfidf[c.key] = tf * idf * boost

        # Normalize by max over candidates (max-scaling, range 0..1)
        max_raw = max(raw_tfidf.values()) if raw_tfidf else 1.0
        if max_raw <= 0:
            max_raw = 1.0
            
        normalized_tfidf = {k: round(v / max_raw, 4) for k, v in raw_tfidf.items()}
        return normalized_tfidf
