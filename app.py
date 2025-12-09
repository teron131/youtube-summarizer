"""
YouTube Summarizer FastAPI Application
=====================================

Optimized FastAPI application for YouTube video processing and summarization.
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from log_config import setup_logging
from models.requests import (
    ConfigurationResponse,
    ScrapResponse,
    SummarizeRequest,
    SummarizeResponse,
    YouTubeRequest,
)

# Import route handlers
from routes.health import (
    get_configuration_response,
    get_health_response,
    get_root_response,
)
from routes.scrap import scrap_video_handler
from routes.summarize import stream_summarize_handler, summarize_handler

# Initialize logging
setup_logging()
load_dotenv()

API_VERSION = "3.0.0"
API_TITLE = "YouTube Summarizer API"
API_DESCRIPTION = "Optimized YouTube video processing and summarization"


# ================================
# FASTAPI APPLICATION SETUP
# ================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"📨 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    return response


# ================================
# API ENDPOINTS
# ================================


@app.get("/")
async def root():
    """API information and health check."""
    return get_root_response()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return get_health_response()


@app.get("/config", response_model=ConfigurationResponse)
async def get_configuration():
    """Get available models and languages."""
    return get_configuration_response()


@app.post("/scrap", response_model=ScrapResponse)
async def scrap_video(request: YouTubeRequest):
    """Extract video metadata and transcript."""
    return await scrap_video_handler(request)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate AI analysis using LangGraph workflow."""
    return await summarize_handler(request)


@app.post("/stream-summarize")
async def stream_summarize(request: SummarizeRequest):
    """Stream analysis with real-time progress updates."""
    return await stream_summarize_handler(request)


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    logging.info(f"🚀 Starting {API_TITLE} v{API_VERSION} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
