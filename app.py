"""YouTube Summarizer FastAPI Application"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from log_config import setup_logging
from routes import api_router

setup_logging()
load_dotenv()

API_VERSION = "3.0.0"
API_TITLE = "YouTube Summarizer API"
API_DESCRIPTION = "Optimized YouTube video processing and summarization"

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"📨 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    return response


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    logging.info(f"🚀 Starting {API_TITLE} v{API_VERSION} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
