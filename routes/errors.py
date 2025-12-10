"""Centralized error handling utilities"""

import logging
from fastapi import HTTPException


class ErrorType:
    """Error type constants"""
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_INPUT = "invalid_input"
    MISSING_CONFIG = "missing_config"
    PROCESSING_FAILED = "processing_failed"


def create_http_error(status_code: int, detail: str, error_type: str | None = None) -> HTTPException:
    """Create HTTPException with logging"""
    emoji = "❌"
    if status_code == 400:
        emoji = "⚠️"
    elif status_code == 429:
        emoji = "🚫"
    elif status_code == 500:
        emoji = "💥"

    logging.error(f"{emoji} {error_type or 'Error'}: {detail}")
    return HTTPException(status_code=status_code, detail=detail)


def handle_exception(e: Exception, context: str = "Processing") -> HTTPException:
    """Convert exception to HTTPException with appropriate status code"""
    error_msg = str(e).lower()

    # Quota errors
    if "quota" in error_msg or "rate limit" in error_msg:
        return create_http_error(429, "API quota exceeded", ErrorType.QUOTA_EXCEEDED)

    # Invalid input errors
    if any(keyword in error_msg for keyword in ["400", "invalid", "bad request", "not found"]):
        return create_http_error(400, f"Invalid input: {str(e)[:100]}", ErrorType.INVALID_INPUT)

    # Default server error
    return create_http_error(500, f"{context} failed: {str(e)[:100]}", ErrorType.PROCESSING_FAILED)


def require_env_key(key_name: str) -> None:
    """Raise HTTPException if environment key is missing"""
    import os
    if not os.getenv(key_name):
        raise create_http_error(500, f"Config missing: {key_name}", ErrorType.MISSING_CONFIG)
