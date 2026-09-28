import re
from typing import List, Tuple, Optional, Set

class Summarizer:
    DEF_PATTERN = re.compile(
        r'\b(is\s+(an?|the|a\s+type\s+of|a\s+branch\s+of)|refers\s+to|is\s+defined\s+as|means|consists\s+of)\b',
        re.IGNORECASE
    )

    @classmethod
    def is_definition(cls, sent_text: str, topic_key: str) -> bool:
        match = cls.DEF_PATTERN.search(sent_text)
        if match:
            # Check if topic appears before the definition trigger
            idx = sent_text.lower().find(topic_key)
            if idx != -1 and idx < match.start():
                return True
        return False

    @classmethod
    def generate_summary(
        cls,
        topic_key: str,
        sentences: List[str],
        high_topic_keys: Set[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (summary, definition_sentence)
        Summary is strictly an extractive substring of source text, <= 280 characters.
        """
        candidate_sents: List[Tuple[float, int, str, bool]] = []
        words = topic_key.split()

        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            # Check if all words of topic_key are in sentence (or topic_key is substring)
            if topic_key in sent_lower or all(w in sent_lower for w in words):
                tokens = sent.split()
                token_count = len(tokens)
                
                is_def = cls.is_definition(sent, topic_key)
                
                # Check if topic is in first 5 tokens
                first_5_text = " ".join(tokens[:5]).lower()
                topic_in_first_5 = 1.0 if (topic_key in first_5_text or words[0] in first_5_text) else 0.0
                
                # Other high topics in sentence
                other_high = sum(1 for ht in high_topic_keys if ht != topic_key and ht in sent_lower)
                
                score = 3.0 * (1.0 if is_def else 0.0)
                score += 1.0 * topic_in_first_5
                score += 0.5 * min(1.0, other_high / 2.0)
                score += 0.5 * (1.0 if (8 <= token_count <= 40) else 0.0)
                score -= 1.0 * (1.0 if token_count > 60 else 0.0)
                
                candidate_sents.append((score, idx, sent, is_def))

        if not candidate_sents:
            # Fallback: find any sentence with partial match or first sentence
            for idx, sent in enumerate(sentences):
                if any(w in sent.lower() for w in words):
                    candidate_sents.append((0.0, idx, sent, False))
                    break
                    
        if not candidate_sents and sentences:
            candidate_sents.append((0.0, 0, sentences[0], False))
            
        if not candidate_sents:
            return ("", None)

        # Sort candidate sentences by score DESC, then earlier index ASC
        candidate_sents.sort(key=lambda x: (-x[0], x[1]))
        top_score, top_idx, best_sent, is_def = candidate_sents[0]
        
        definition_sent = best_sent if is_def else None
        
        # If best_sent is short (< 120 chars) and next sentence exists, try appending
        summary = best_sent.strip()
        if len(summary) < 120 and top_idx + 1 < len(sentences):
            next_sent = sentences[top_idx + 1].strip()
            combined = summary + " " + next_sent
            if len(combined) <= 280:
                summary = combined

        # Enforce <= 280 chars boundary
        if len(summary) > 280:
            summary = summary[:277].rstrip() + "..."
            
        return (summary, definition_sent)
