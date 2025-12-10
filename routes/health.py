"""Health check and configuration routes"""

import os
from datetime import datetime

from youtube_summarizer.summarizer import ANALYSIS_MODEL, QUALITY_MODEL, TARGET_LANGUAGE

API_VERSION = "3.0.0"
API_TITLE = "YouTube Summarizer API"

AVAILABLE_MODELS = {
    "x-ai/grok-4.1-fast": "Grok 4.1 Fast (Recommended)",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash (Fast)",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
}

SUPPORTED_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "ru": "Russian",
}


def get_health_response():
    return {
        "status": "healthy",
        "message": f"{API_TITLE} is running",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION,
        "environment": {
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "scrapecreators_configured": bool(os.getenv("SCRAPECREATORS_API_KEY")),
        },
    }


def get_configuration_response():
    return {
        "status": "success",
        "message": "Configuration retrieved successfully",
        "available_models": AVAILABLE_MODELS,
        "supported_languages": SUPPORTED_LANGUAGES,
        "default_analysis_model": ANALYSIS_MODEL,
        "default_quality_model": QUALITY_MODEL,
        "default_target_language": TARGET_LANGUAGE,
    }


def get_root_response():
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
            "POST /scrap": "Extract video metadata and transcript",
            "POST /summarize": "Full LangGraph workflow analysis",
            "POST /stream-summarize": "Streaming analysis with progress",
        },
        "timestamp": datetime.now().isoformat(),
    }
