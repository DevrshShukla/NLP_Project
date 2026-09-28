import math
from typing import Dict, List, Optional
from backend.nlp.ngram_extractor import Candidate
from backend.nlp.priority_classifier import PriorityClassifier

class ScoredTopic:
    def __init__(
        self,
        candidate: Candidate,
        tfidf_c: float,
        frequency_c: float,
        position_c: float,
        semantic_c: Optional[float],
        score: float,
        priority: str,
        first_position_pct: float
    ):
        self.candidate = candidate
        self.tfidf_c = tfidf_c
        self.frequency_c = frequency_c
        self.position_c = position_c
        self.semantic_c = semantic_c
        self.score = score
        self.priority = priority
        self.first_position_pct = first_position_pct
        self.rank = 0

class TopicRanker:
    @classmethod
    def rank_topics(
        cls,
        candidates: List[Candidate],
        tfidf_scores: Dict[str, float],
        doc_length: int,
        weights: Dict[str, float],
        high_threshold: float = 80.0,
        medium_threshold: float = 50.0,
        semantic_scores: Optional[Dict[str, float]] = None,
        max_topics: int = 25
    ) -> List[ScoredTopic]:
        if not candidates:
            return []
            
        max_count = max(c.count for c in candidates) if candidates else 1
        denom_freq = math.log(1 + max_count) if max_count > 0 else 1.0
        if denom_freq <= 0:
            denom_freq = 1.0
            
        # Determine enabled signals & sum of enabled weights
        w_tfidf = weights.get("tfidf", 0.40)
        w_freq = weights.get("frequency", 0.25)
        w_pos = weights.get("position", 0.20)
        w_sem = weights.get("semantic", 0.15) if semantic_scores is not None else 0.0
        
        sum_weights = w_tfidf + w_freq + w_pos + w_sem
        if sum_weights <= 0:
            sum_weights = 1.0

        scored_topics: List[ScoredTopic] = []
        
        for c in candidates:
            tfidf_c = tfidf_scores.get(c.key, 0.0)
            
            # frequency_c: ln(1 + count) / ln(1 + max_count)
            frequency_c = round(math.log(1 + c.count) / denom_freq, 4)
            
            # position_c: 0.5 * earliness + 0.3 * heading_flag + 0.2 * definition_flag
            # earliness = 1 - min(1, first_char / min(len, 20000))
            char_cap = max(1, min(doc_length, 20000))
            earliness = 1.0 - min(1.0, c.first_char / char_cap)
            heading_flag = 1.0 if c.in_heading else 0.0
            definition_flag = 1.0 if c.in_definition else 0.0
            position_c = round(0.5 * earliness + 0.3 * heading_flag + 0.2 * definition_flag, 4)
            
            # semantic_c
            semantic_c = None
            sem_term = 0.0
            if semantic_scores is not None:
                sem_val = semantic_scores.get(c.key, 0.0)
                semantic_c = round(sem_val, 4)
                sem_term = w_sem * semantic_c
                
            # Score formula: 100 * (w_tfidf*tfidf_c + w_freq*frequency_c + w_pos*position_c + w_sem*semantic_c) / sum_weights
            raw_score = 100.0 * (
                w_tfidf * tfidf_c +
                w_freq * frequency_c +
                w_pos * position_c +
                sem_term
            ) / sum_weights
            
            score = round(raw_score, 1)
            score = max(0.0, min(100.0, score))
            
            priority = PriorityClassifier.classify(score, high_threshold, medium_threshold)
            first_pos_pct = round(c.first_char / max(1, doc_length), 4)
            
            scored_topics.append(ScoredTopic(
                candidate=c,
                tfidf_c=tfidf_c,
                frequency_c=frequency_c,
                position_c=position_c,
                semantic_c=semantic_c,
                score=score,
                priority=priority,
                first_position_pct=first_pos_pct
            ))

        # Tie-break: higher score DESC, higher occurrences DESC, alphabetical by normalized key ASC
        scored_topics.sort(
            key=lambda t: (-t.score, -t.candidate.count, t.candidate.key)
        )
        
        # Filter: keep topics with score >= 20.0 (or top 5 if all below 20)
        filtered = [t for t in scored_topics if t.score >= 20.0]
        if len(filtered) < 5 and len(scored_topics) > 0:
            filtered = scored_topics[:min(5, len(scored_topics))]
            
        final_list = filtered[:max_topics]
        for idx, t in enumerate(final_list):
            t.rank = idx + 1
            
        return final_list
