import math
import re
from typing import List, Dict, Set, Tuple, Optional
import spacy

class Candidate:
    def __init__(
        self,
        key: str,
        surface: str,
        ngram_size: int,
        count: int,
        first_char: int,
        section_ids: Set[int],
        in_heading: bool = False,
        in_definition: bool = False
    ):
        self.key = key
        self.surface = surface
        self.ngram_size = ngram_size
        self.count = count
        self.first_char = first_char
        self.section_ids = section_ids
        self.in_heading = in_heading
        self.in_definition = in_definition
        self.length_boost = 1.0 if ngram_size == 1 else (1.15 if ngram_size == 2 else 1.25)
        self.occurrences_spans: List[Tuple[int, int]] = []

class NgramExtractor:
    # Accept NOUN, PROPN, ADJ, and participial modifiers (VBN, VBG)
    NOUN_POS = {"NOUN", "PROPN"}
    PARTICIPLE_TAGS = {"VBN", "VBG"}
    
    DEF_PATTERN = re.compile(
        r'\b(is\s+(an?|the|a\s+type\s+of|a\s+branch\s+of)|refers\s+to|is\s+defined\s+as|means|uses?|predicts?)\b',
        re.IGNORECASE
    )

    @classmethod
    def is_valid_token(cls, token: spacy.tokens.Token) -> bool:
        if token.is_punct or token.is_space or token.is_stop or token.pos_ == "DET":
            return False
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
            return True
        if token.tag_ in cls.PARTICIPLE_TAGS:
            return True
        return False

    @classmethod
    def clean_chunk(cls, chunk: spacy.tokens.Span, academic_stopwords: Set[str]) -> Optional[Tuple[str, str, int, int, int]]:
        tokens = [t for t in chunk if not t.is_space and not t.is_punct and t.pos_ != "DET"]
        # Strip leading/trailing stopwords or academic fillers
        while tokens and (tokens[0].is_stop or tokens[0].lemma_.lower() in academic_stopwords):
            tokens.pop(0)
        while tokens and (tokens[-1].is_stop or tokens[-1].lemma_.lower() in academic_stopwords):
            tokens.pop()
            
        if not tokens or len(tokens) > 3:
            return None
        # Last token must be a noun
        if tokens[-1].pos_ not in cls.NOUN_POS:
            return None
        # All tokens must be valid
        if not all(cls.is_valid_token(t) for t in tokens):
            return None
            
        key = " ".join(t.lemma_.lower() for t in tokens)
        surface = " ".join(t.text for t in tokens)
        if len(surface) > 1 and surface[0].islower():
            surface = surface.title()
            
        return key, surface, len(tokens), tokens[0].idx, tokens[-1].idx + len(tokens[-1].text)

    @classmethod
    def extract_candidates(
        cls,
        doc: spacy.tokens.Doc,
        section_id: int,
        academic_stopwords: Set[str],
        is_heading_section: bool = False
    ) -> List[Candidate]:
        candidates: Dict[str, Candidate] = {}

        # 1. Extract from spaCy Noun Chunks
        for chunk in doc.noun_chunks:
            res = cls.clean_chunk(chunk, academic_stopwords)
            if not res:
                continue
            key, surface, n_size, start_ch, end_ch = res
            
            # Check definition subject
            sent_text = chunk.sent.text
            has_def = False
            def_match = cls.DEF_PATTERN.search(sent_text)
            if def_match and (start_ch - chunk.sent.start_char) < def_match.start():
                has_def = True

            in_head = is_heading_section
            
            if key not in candidates:
                candidates[key] = Candidate(
                    key=key,
                    surface=surface,
                    ngram_size=n_size,
                    count=1,
                    first_char=start_ch,
                    section_ids={section_id},
                    in_heading=in_head,
                    in_definition=has_def
                )
                candidates[key].occurrences_spans.append((start_ch, end_ch))
            else:
                c = candidates[key]
                c.count += 1
                c.section_ids.add(section_id)
                c.in_heading = c.in_heading or in_head
                c.in_definition = c.in_definition or has_def
                c.occurrences_spans.append((start_ch, end_ch))

        # 2. Extract N-grams clause-by-clause (never crossing punctuation like commas or semicolons)
        for sent in doc.sents:
            # Split sentence into punctuation-separated clauses
            clauses: List[List[spacy.tokens.Token]] = []
            current_clause: List[spacy.tokens.Token] = []
            
            for token in sent:
                if token.is_punct:
                    if current_clause:
                        clauses.append(current_clause)
                        current_clause = []
                else:
                    if not token.is_space:
                        current_clause.append(token)
            if current_clause:
                clauses.append(current_clause)

            for clause in clauses:
                n_tokens = len(clause)
                for n in range(1, 4):
                    for i in range(n_tokens - n + 1):
                        window = clause[i : i + n]
                        # Last token must be noun
                        if window[-1].pos_ not in cls.NOUN_POS:
                            continue
                        # All tokens must be valid
                        if not all(cls.is_valid_token(t) for t in window):
                            continue
                        if window[0].is_stop or window[-1].is_stop:
                            continue
                        if window[0].lemma_.lower() in academic_stopwords or window[-1].lemma_.lower() in academic_stopwords:
                            continue
                        if any(t.text.isdigit() for t in window):
                            continue
                            
                        key = " ".join(t.lemma_.lower() for t in window)
                        surface = " ".join(t.text for t in window)
                        if len(surface) > 1 and surface[0].islower():
                            surface = surface.title()
                            
                        start_ch = window[0].idx
                        end_ch = window[-1].idx + len(window[-1].text)
                        
                        has_def = False
                        def_match = cls.DEF_PATTERN.search(sent.text)
                        if def_match and (start_ch - sent.start_char) < def_match.start():
                            has_def = True
                            
                        if key not in candidates:
                            candidates[key] = Candidate(
                                key=key,
                                surface=surface,
                                ngram_size=n,
                                count=1,
                                first_char=start_ch,
                                section_ids={section_id},
                                in_heading=is_heading_section,
                                in_definition=has_def
                            )
                            candidates[key].occurrences_spans.append((start_ch, end_ch))
                        else:
                            c = candidates[key]
                            if (start_ch, end_ch) not in c.occurrences_spans:
                                c.count += 1
                                c.occurrences_spans.append((start_ch, end_ch))
                                c.section_ids.add(section_id)
                                c.in_definition = c.in_definition or has_def

        return list(candidates.values())

    @classmethod
    def apply_filters_and_subsumption(
        cls,
        candidates: List[Candidate],
        total_tokens: int
    ) -> List[Candidate]:
        if not candidates:
            return []

        # Filter out candidates with count < 2 if total_tokens >= 1000 and not in heading/definition
        valid_candidates = []
        for c in candidates:
            if total_tokens >= 1000 and c.ngram_size >= 2:
                if c.count < 2 and not (c.in_heading or c.in_definition):
                    continue
            valid_candidates.append(c)

        # Subsumption: Drop shorter candidate S if >= 80% of its occurrences are inside kept longer candidates L
        longer_candidates = [c for c in valid_candidates if c.ngram_size >= 2]
        
        final_candidates: List[Candidate] = []
        for c in valid_candidates:
            if c.ngram_size < 3:
                subsumed_count = 0
                c_words = c.key.split()
                for longer in longer_candidates:
                    if longer.ngram_size > c.ngram_size:
                        longer_words = longer.key.split()
                        # check subphrase match
                        for idx in range(len(longer_words) - len(c_words) + 1):
                            if longer_words[idx : idx + len(c_words)] == c_words:
                                subsumed_count += longer.count
                                break
                                
                if c.count > 0 and (subsumed_count / c.count) >= 0.80 and not c.in_heading:
                    # Dropped by subsumption!
                    continue

            final_candidates.append(c)

        return final_candidates
