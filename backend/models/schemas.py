from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ComponentScores(BaseModel):
    tfidf: float
    frequency: float
    position: float
    semantic: Optional[float] = None

class TopicItem(BaseModel):
    id: str
    name: str
    normalized: str
    rank: int
    score: float
    priority: str
    ngram_size: int
    occurrences: int
    first_position_pct: float
    components: ComponentScores
    summary: str
    definition: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    cluster_id: Optional[str] = None

class KeywordItem(BaseModel):
    term: str
    score: float
    count: int

class DocumentMeta(BaseModel):
    id: Optional[str] = None
    title: str = "Untitled document"
    source_type: str = "text"
    char_count: int
    word_count: int
    section_count: int
    language: str = "en"
    page_count: Optional[int] = None
    extraction_quality: Optional[float] = None

class StatsMeta(BaseModel):
    topics_found: int
    high: int
    medium: int
    low: int
    processing_ms: int

class WeightsConfig(BaseModel):
    tfidf: float = 0.40
    frequency: float = 0.25
    position: float = 0.20
    semantic: float = 0.15

class ThresholdsConfig(BaseModel):
    high: float = 80.0
    medium: float = 50.0

class ConfigMeta(BaseModel):
    weights: Dict[str, float]
    thresholds: ThresholdsConfig
    semantic_enabled: bool = False
    max_topics: int = 25
    pipeline_version: str = "1.0.0"

class GraphNode(BaseModel):
    id: str
    label: str
    score: float
    priority: str
    cluster_id: Optional[str] = None
    depth: int = 0

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str  # "hierarchy", "semantic", "cooccurrence"
    weight: float

class RelationshipGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class HierarchyNode(BaseModel):
    id: str
    name: str
    score: float
    priority: str
    depth: int
    children: List["HierarchyNode"] = Field(default_factory=list)

class AnalysisResult(BaseModel):
    analysis_id: str
    document: DocumentMeta
    stats: StatsMeta
    config: ConfigMeta
    topics: List[TopicItem]
    keywords: List[KeywordItem]
    hierarchy: Optional[List[HierarchyNode]] = None
    graph: Optional[RelationshipGraph] = None
    warnings: List[str] = Field(default_factory=list)

class AnalyzeOptions(BaseModel):
    max_topics: Optional[int] = 25
    weights: Optional[Dict[str, float]] = None
    thresholds: Optional[ThresholdsConfig] = None
    semantic_enabled: Optional[bool] = False
    include_hierarchy: Optional[bool] = True
    include_graph: Optional[bool] = True

class AnalyzeRequest(BaseModel):
    text: str
    title: Optional[str] = "Untitled document"
    options: Optional[AnalyzeOptions] = None

class UploadResponse(BaseModel):
    text: str
    page_count: int
    quality: float
    char_count: int
    word_count: int
    title: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

class ExportOptions(BaseModel):
    include_summaries: bool = True
    include_keywords: bool = True
    priority_filter: Optional[str] = "ALL"  # "ALL", "HIGH", "MEDIUM", "LOW"

class ExportRequest(BaseModel):
    analysis: AnalysisResult
    format: str  # "pdf", "csv", "txt", "md", "json"
    options: Optional[ExportOptions] = None

class QuestionItem(BaseModel):
    id: str
    topic_id: str
    topic_name: str
    question_text: str
    question_type: str
    difficulty: str
    generation_method: str = "template"
    source_sentence: Optional[str] = None

class QuestionsResponse(BaseModel):
    questions: List[QuestionItem]
    total: int
