import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from backend.core.config import settings
from backend.core.errors import AppError, create_app_error
from backend.core.logging import logger
from backend.core.rate_limit import limiter
from backend.models.schemas import (
    AnalyzeRequest,
    AnalysisResult,
    UploadResponse,
    ExportRequest,
    QuestionsResponse
)
from backend.nlp.pipeline import NLPPipeline
from backend.nlp.question_generator import QuestionGenerator
from backend.services.pdf_processor import PDFProcessor
from backend.services.exporter import Exporter

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize pipeline once
    logger.info("Initializing Study Topic Extractor backend...")
    app.state.pipeline = NLPPipeline()
    logger.info("Study Topic Extractor backend is ready.")
    yield
    # Shutdown
    logger.info("Shutting down Study Topic Extractor backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PIPELINE_VERSION,
    lifespan=lifespan
)

# Attach SlowAPI limiter
app.state.limiter = limiter

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# Request ID & Logging Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id
    start_time = time.time()
    
    try:
        response: Response = await call_next(request)
        latency_ms = int((time.time() - start_time) * 1000)
        response.headers["X-Request-ID"] = req_id
        
        # Log without document text
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({latency_ms}ms)",
            extra={
                "request_id": req_id,
                "route": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms
            }
        )
        return response
    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
            extra={
                "request_id": req_id,
                "route": request.url.path,
                "latency_ms": latency_ms
            }
        )
        raise exc

# Global Exception Handlers conforming to specification envelope
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:12]}")
    return JSONResponse(
        status_code=exc.status_code,
        headers={"X-Request-ID": req_id},
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": req_id
            }
        }
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:12]}")
    app_err = create_app_error("RATE_LIMITED")
    return JSONResponse(
        status_code=app_err.status_code,
        headers={"X-Request-ID": req_id, "Retry-After": "60"},
        content={
            "error": {
                "code": app_err.code,
                "message": app_err.message,
                "details": {"retry_after": 60},
                "request_id": req_id
            }
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:12]}")
    app_err = create_app_error("PROCESSING_FAILED")
    return JSONResponse(
        status_code=app_err.status_code,
        headers={"X-Request-ID": req_id},
        content={
            "error": {
                "code": app_err.code,
                "message": app_err.message,
                "details": {},
                "request_id": req_id
            }
        }
    )

# ----------------- API Endpoints -----------------

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.PIPELINE_VERSION,
        "models_loaded": {
            "spacy": hasattr(app.state, "pipeline") and app.state.pipeline.nlp is not None,
            "background_idf": hasattr(app.state, "pipeline") and len(app.state.pipeline.tfidf_service.bg_terms) > 0
        }
    }

@app.post("/api/upload", response_model=UploadResponse)
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...)
):
    # Validate content type and filename
    if not file.filename.lower().endswith(".pdf"):
        raise create_app_error("INVALID_FILE_TYPE")
        
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise create_app_error("INVALID_PDF", "The uploaded file is empty.")

    # Process in memory
    result = PDFProcessor.process_pdf_bytes(
        pdf_bytes=pdf_bytes,
        max_mb=settings.MAX_PDF_MB,
        max_pages=settings.MAX_PDF_PAGES
    )
    
    # Deriving title from filename
    raw_name = file.filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    title = raw_name if len(raw_name) <= 120 else raw_name[:117] + "..."

    return UploadResponse(
        text=result["text"],
        page_count=result["page_count"],
        quality=result["quality"],
        char_count=result["char_count"],
        word_count=result["word_count"],
        title=title,
        warnings=result["warnings"]
    )

@app.post("/api/analyze", response_model=AnalysisResult)
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def analyze_document(
    request: Request,
    payload: AnalyzeRequest
):
    text = payload.text
    if not text or len(text.strip()) < settings.MIN_TEXT_CHARS:
        raise create_app_error("TEXT_TOO_SHORT")
    if len(text) > settings.MAX_TEXT_CHARS:
        raise create_app_error("DOCUMENT_TOO_LARGE")

    # Run the pipeline
    pipeline: NLPPipeline = app.state.pipeline
    result = pipeline.run(
        text=text,
        title=payload.title or "Untitled document",
        source_type="text",
        options=payload.options
    )
    return result

@app.post("/api/export")
async def export_roadmap(
    request: Request,
    payload: ExportRequest
):
    file_bytes, mime_type, filename = Exporter.export(
        analysis=payload.analysis,
        export_format=payload.format,
        options=payload.options
    )
    return Response(
        content=file_bytes,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/api/documents/{id}/questions", response_model=QuestionsResponse)
async def generate_questions_for_topics(
    id: str,
    payload: AnalysisResult
):
    # Generate on-demand questions from topics & hierarchy
    scored_topics = []
    # Reconstruct ScoredTopics for generator
    definitions_map = {}
    parent_map = {}
    children_map = {}
    
    for t in payload.topics:
        definitions_map[t.normalized] = t.definition
        parent_map[t.normalized] = t.parent_id
        children_map[t.normalized] = t.children_ids

    # Mock lightweight scored topic objects
    class MockCandidate:
        def __init__(self, key, surface, count):
            self.key = key
            self.surface = surface
            self.count = count

    class MockTopic:
        def __init__(self, candidate, score, priority):
            self.candidate = candidate
            self.score = score
            self.priority = priority

    mock_topics = [
        MockTopic(MockCandidate(t.normalized, t.name, t.occurrences), t.score, t.priority)
        for t in payload.topics
    ]

    questions = QuestionGenerator.generate_questions(
        topics=mock_topics,
        parent_map=parent_map,
        children_map=children_map,
        definitions_map=definitions_map,
        only_high=False
    )

    return QuestionsResponse(
        questions=questions,
        total=len(questions)
    )
