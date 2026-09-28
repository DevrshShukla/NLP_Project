import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "route"):
            log_obj["route"] = record.route
        if hasattr(record, "status_code"):
            log_obj["status_code"] = record.status_code
        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = record.latency_ms
        if hasattr(record, "char_count"):
            log_obj["char_count"] = record.char_count
        if hasattr(record, "page_count"):
            log_obj["page_count"] = record.page_count
        if hasattr(record, "error_code"):
            log_obj["error_code"] = record.error_code
        return json.dumps(log_obj)

def setup_logger(name: str = "study_extractor") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()
