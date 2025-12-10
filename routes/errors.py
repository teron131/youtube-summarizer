"""Centralized error handling utilities"""

import logging
import os

from fastapi import HTTPException

EMOJI_MAP = {
    400: "⚠️",
    429: "🚫",
    500: "💥",
}


def create_http_error(
    status_code: int,
    detail: str,
    error_type: str | None = None,
) -> HTTPException:
    emoji = EMOJI_MAP.get(status_code, "❌")
    logging.error(f"{emoji} {error_type or 'Error'}: {detail}")
    return HTTPException(status_code=status_code, detail=detail)


def handle_exception(e: Exception, context: str = "Processing") -> HTTPException:
    error_msg = str(e).lower()

    if "quota" in error_msg or "rate limit" in error_msg:
        return create_http_error(429, "API quota exceeded", "quota_exceeded")

    if any(kw in error_msg for kw in ["400", "invalid", "bad request", "not found"]):
        return create_http_error(400, f"Invalid input: {str(e)[:100]}", "invalid_input")

    return create_http_error(500, f"{context} failed: {str(e)[:100]}", "processing_failed")


def require_env_key(key_name: str) -> None:
    if not os.getenv(key_name):
        raise create_http_error(500, f"Config missing: {key_name}", "missing_config")
