import time
import uuid
import json
import os
from typing import Dict, List, Optional, Set, Tuple
import spacy

from backend.core.config import settings
from backend.core.logging import logger
from backend.models.schemas import (
    AnalysisResult,
    DocumentMeta,
    StatsMeta,
    ConfigMeta,
    TopicItem,
    ComponentScores,
    KeywordItem,
    ThresholdsConfig,
    AnalyzeOptions
)
from backend.nlp.text_cleaner import TextCleaner
from backend.nlp.sectioner import Sectioner
from backend.nlp.ngram_extractor import NgramExtractor, Candidate
from backend.nlp.tfidf_service import TfidfService
from backend.nlp.topic_ranker import TopicRanker, ScoredTopic
from backend.nlp.summarizer import Summarizer
from backend.nlp.hierarchy_builder import HierarchyBuilder
from backend.nlp.graph_builder import GraphBuilder

class NLPPipeline:
    def __init__(self):
        # Load spaCy model
        logger.info("Loading spaCy model en_core_web_sm...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load academic stopwords
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stopwords_path = os.path.join(base_dir, "data", "stopwords_academic.json")
        self.academic_stopwords: Set[str] = set()
        if os.path.exists(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as f:
                self.academic_stopwords = set(json.load(f))
                
        # Load background IDF service
        bg_idf_path = os.path.join(base_dir, "data", "background_idf.json")
        self.tfidf_service = TfidfService(bg_idf_path)
        logger.info("NLP Pipeline initialized successfully.")

    def run(
        self,
        text: str,
        title: str = "Untitled document",
        source_type: str = "text",
        page_count: Optional[int] = None,
        extraction_quality: Optional[float] = None,
        options: Optional[AnalyzeOptions] = None
    ) -> AnalysisResult:
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        warnings: List[str] = []

        char_count = len(text)
        word_count = len(text.split())

        if char_count < 600:
            warnings.append("SHORT_DOCUMENT")

        # 1. Clean Text
        cleaned_text = TextCleaner.clean(text)
        
        # Check math symbols ratio
        symbol_count = sum(1 for ch in cleaned_text if ch in "+-*/=<>^%&|~#")
        if len(cleaned_text) > 0 and (symbol_count / len(cleaned_text)) > 0.05:
            warnings.append("MATH_CONTENT")

        # 2. Section Document
        sections = Sectioner.split_into_sections(cleaned_text)
        section_count = len(sections)

        # 3. Preprocess with spaCy & Extract candidates per section
        all_candidates_map: Dict[str, Candidate] = {}
        all_sentences: List[str] = []
        total_tokens = 0

        for sec in sections:
            doc = self.nlp(sec.text)
            total_tokens += len(doc)
            
            # Record sentences
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) > 5:
                    all_sentences.append(sent_text)
                    
            # Extract n-gram candidates in section
            is_head = bool(sec.heading)
            sec_candidates = NgramExtractor.extract_candidates(
                doc=doc,
                section_id=sec.index,
                academic_stopwords=self.academic_stopwords,
                is_heading_section=is_head
            )
            
            # Merge candidates globally
            for c in sec_candidates:
                if c.key not in all_candidates_map:
                    all_candidates_map[c.key] = c
                else:
                    existing = all_candidates_map[c.key]
                    existing.count += c.count
                    existing.section_ids.update(c.section_ids)
                    existing.in_heading = existing.in_heading or c.in_heading
                    existing.in_definition = existing.in_definition or c.in_definition
                    existing.occurrences_spans.extend(c.occurrences_spans)
                    if c.surface[0].isupper() and not existing.surface[0].isupper():
                        existing.surface = c.surface

        # Deduplicate sentences
        deduped_sentences_pairs = TextCleaner.deduplicate_sentences(all_sentences)
        sentences = [p[0] for p in deduped_sentences_pairs]

        # 4. Apply N-gram filters and subsumption
        filtered_candidates = NgramExtractor.apply_filters_and_subsumption(
            list(all_candidates_map.values()),
            total_tokens=total_tokens
        )

        # 5. Compute TF-IDF
        tfidf_scores = self.tfidf_service.compute_tfidf(
            candidates=filtered_candidates,
            total_sections=section_count
        )

        # 6. Extract Document-level Keywords (Top 30 by TF-IDF * length boost)
        keyword_candidates = [
            c for c in filtered_candidates
            if c.surface.lower() not in self.academic_stopwords
        ]
        keyword_candidates.sort(key=lambda c: -tfidf_scores.get(c.key, 0.0))
        top_keywords: List[KeywordItem] = []
        for c in keyword_candidates[:30]:
            top_keywords.append(KeywordItem(
                term=c.surface,
                score=tfidf_scores.get(c.key, 0.0),
                count=c.count
            ))

        # 7. Topic Ranking & Priority Classification
        # Resolve config & weights
        user_opts = options or AnalyzeOptions()
        weights_dict = {
            "tfidf": settings.WEIGHT_TFIDF,
            "frequency": settings.WEIGHT_FREQUENCY,
            "position": settings.WEIGHT_POSITION,
            "semantic": settings.WEIGHT_SEMANTIC if (user_opts.semantic_enabled or settings.ENABLE_SEMANTIC) else 0.0
        }
        if user_opts.weights:
            for k, v in user_opts.weights.items():
                if k in weights_dict:
                    weights_dict[k] = v
                    
        thresholds = user_opts.thresholds or ThresholdsConfig(
            high=settings.PRIORITY_HIGH_MIN,
            medium=settings.PRIORITY_MEDIUM_MIN
        )
        max_topics = user_opts.max_topics or settings.MAX_TOPICS

        scored_topics = TopicRanker.rank_topics(
            candidates=filtered_candidates,
            tfidf_scores=tfidf_scores,
            doc_length=len(cleaned_text),
            weights=weights_dict,
            high_threshold=thresholds.high,
            medium_threshold=thresholds.medium,
            semantic_scores=None,
            max_topics=max_topics
        )

        if not scored_topics:
            warnings.append("NO_TOPICS_FOUND")

        # 8. Topic Hierarchy & Graph
        high_topic_keys = {t.candidate.key for t in scored_topics if t.priority == "HIGH"}
        
        parent_map, children_map, depth_map, hierarchy_roots = HierarchyBuilder.build_hierarchy(
            topics=scored_topics,
            sentences=sentences,
            total_sections=section_count
        )
        
        graph = GraphBuilder.build_graph(
            topics=scored_topics,
            parent_map=parent_map,
            depth_map=depth_map,
            sentences=sentences
        )

        # 9. Topic Summaries & Definitions & Per-Topic Keywords
        final_topics: List[TopicItem] = []
        for t in scored_topics:
            k = t.candidate.key
            summary_text, def_text = Summarizer.generate_summary(
                topic_key=k,
                sentences=sentences,
                high_topic_keys=high_topic_keys
            )
            
            # Find per-topic keywords (co-occurring candidates)
            topic_kw = []
            words_in_t = set(k.split())
            for kw in top_keywords:
                kw_lower = kw.term.lower()
                if kw_lower != k and kw_lower not in words_in_t:
                    # check if kw co-occurs in sentences with this topic
                    for sent in sentences:
                        s_lower = sent.lower()
                        if k in s_lower and kw_lower in s_lower:
                            topic_kw.append(kw.term)
                            break
                if len(topic_kw) >= 4:
                    break

            final_topics.append(TopicItem(
                id=f"t_{t.rank:03d}",
                name=t.candidate.surface,
                normalized=t.candidate.key,
                rank=t.rank,
                score=t.score,
                priority=t.priority,
                ngram_size=t.candidate.ngram_size,
                occurrences=t.candidate.count,
                first_position_pct=t.first_position_pct,
                components=ComponentScores(
                    tfidf=t.tfidf_c,
                    frequency=t.frequency_c,
                    position=t.position_c,
                    semantic=t.semantic_c
                ),
                summary=summary_text,
                definition=def_text,
                keywords=topic_kw,
                parent_id=parent_map.get(k),
                children_ids=children_map.get(k, []),
                cluster_id=None
            ))

        # 10. Assemble Result
        elapsed_ms = int((time.time() - start_time) * 1000)
        high_count = sum(1 for t in final_topics if t.priority == "HIGH")
        medium_count = sum(1 for t in final_topics if t.priority == "MEDIUM")
        low_count = sum(1 for t in final_topics if t.priority == "LOW")

        result = AnalysisResult(
            analysis_id=analysis_id,
            document=DocumentMeta(
                id=None,
                title=title,
                source_type=source_type,
                char_count=char_count,
                word_count=word_count,
                section_count=section_count,
                language="en",
                page_count=page_count,
                extraction_quality=extraction_quality
            ),
            stats=StatsMeta(
                topics_found=len(final_topics),
                high=high_count,
                medium=medium_count,
                low=low_count,
                processing_ms=elapsed_ms
            ),
            config=ConfigMeta(
                weights=weights_dict,
                thresholds=thresholds,
                semantic_enabled=user_opts.semantic_enabled or False,
                max_topics=max_topics,
                pipeline_version=settings.PIPELINE_VERSION
            ),
            topics=final_topics,
            keywords=top_keywords,
            hierarchy=hierarchy_roots,
            graph=graph,
            warnings=warnings
        )

        return result
