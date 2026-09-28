from typing import Any, Dict, Optional

class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

ERROR_CATALOG = {
    "INVALID_REQUEST": (400, "The request could not be understood."),
    "INVALID_PDF": (400, "Please upload a valid PDF file."),
    "INVALID_FILE_TYPE": (415, "This file type is not supported. Please upload a PDF."),
    "PASSWORD_PROTECTED_PDF": (422, "This PDF is password protected. Remove the password and try again."),
    "NO_READABLE_TEXT": (422, "We couldn't find readable text in this document."),
    "TEXT_TOO_SHORT": (422, "Please provide at least 200 characters of study material."),
    "UNAUTHORIZED": (401, "Please sign in to continue."),
    "FORBIDDEN": (403, "You don't have access to this resource."),
    "NOT_FOUND": (404, "We couldn't find what you were looking for."),
    "FILE_TOO_LARGE": (413, "This file is larger than the 10 MB upload limit."),
    "DOCUMENT_TOO_LARGE": (413, "This document is larger than the supported processing limit."),
    "RATE_LIMITED": (429, "Too many requests. Please wait a moment and try again."),
    "PROCESSING_FAILED": (500, "We couldn't analyze this document. Please try again."),
    "SERVICE_UNAVAILABLE": (503, "The service is temporarily unavailable. Please try again shortly."),
}

def create_app_error(code: str, custom_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> AppError:
    if code in ERROR_CATALOG:
        status, default_msg = ERROR_CATALOG[code]
        return AppError(code=code, message=custom_message or default_msg, status_code=status, details=details)
    return AppError(code=code, message=custom_message or "An unexpected error occurred.", status_code=500, details=details)
