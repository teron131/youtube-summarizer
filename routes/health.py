"""Health check and configuration endpoints for API monitoring."""

from datetime import UTC, datetime
import os

from fastapi import APIRouter

from routes.schema import ConfigurationResponse
from youtube_summarizer.scrapper.supadata import get_supadata_api_key
from youtube_summarizer.summarizer import QUALITY_MODEL, SUMMARY_MODEL, TARGET_LANGUAGE

router = APIRouter()

API_VERSION = "3.0.0"
API_TITLE = "YouTube Summarizer API"

AVAILABLE_MODELS = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "openai/gpt-5.2": "GPT-5.2",
    "x-ai/grok-4.1-fast": "Grok 4.1 Fast",
    "google/gemini-2.5-flash-lite-preview-09-2025": "Gemini 2.5 Flash Lite",
}

SUPPORTED_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "ru": "Russian",
}


@router.get("/")
async def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "Optimized YouTube video processing and summarization",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check with environment status",
            "GET /config": "Get available models and languages",
            "POST /scrape": "Extract video metadata and transcript",
            "POST /summarize": "Full LangGraph workflow summary",
            "POST /stream-summarize": "Streaming summary with progress",
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": f"{API_TITLE} is running",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": API_VERSION,
        "environment": {
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "scrapecreators_configured": bool(os.getenv("SCRAPECREATORS_API_KEY")),
            "supadata_configured": bool(get_supadata_api_key()),
        },
    }


@router.get("/config", response_model=ConfigurationResponse)
async def get_configuration():
    return {
        "status": "success",
        "message": "Configuration retrieved successfully",
        "available_models": AVAILABLE_MODELS,
        "supported_languages": SUPPORTED_LANGUAGES,
        "default_summary_model": SUMMARY_MODEL,
        "default_quality_model": QUALITY_MODEL,
        "default_target_language": TARGET_LANGUAGE,
    }
