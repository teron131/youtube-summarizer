#!/bin/bash

# Production startup script for YouTube Summarizer Backend API
set -e

echo "🚀 Starting YouTube Summarizer Backend API..."

# Check if we're in Railway (has PORT env var)
if [ -n "$PORT" ]; then
    echo "🚂 Detected Railway deployment"
    HOST=${HOST:-"0.0.0.0"}
    PORT=${PORT}
    WORKERS=${WORKERS:-1}
else
    echo "🏠 Local production mode"
    HOST=${HOST:-"localhost"}
    PORT=${PORT:-8080}
    WORKERS=${WORKERS:-1}
fi

echo "🌍 Starting API server on $HOST:$PORT with $WORKERS workers"

# Use uvicorn with import string for proper deployment
exec python -m uvicorn app:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --access-log