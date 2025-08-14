#!/bin/bash

# YouTube Summarizer Startup Script (Dev/Prod)
echo "🎬 Starting YouTube Summarizer Environment"
echo "Frontend (Next.js): port will be set from $PORT or 3000"
echo "Backend API (FastAPI): http://localhost:8080"
echo ""

# Ensure frontend knows how to reach the backend locally
if [ -z "$BACKEND_URL" ]; then
	export BACKEND_URL="http://127.0.0.1:8080"
fi
echo "🔗 BACKEND_URL=$BACKEND_URL"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Prepare Python runtime (prefer uv to isolate deps)
PYTHON_CMD="python"
if command -v uv >/dev/null 2>&1; then
	echo "🧪 Using uv virtual env (./.venv)"
	uv sync >/dev/null 2>&1 || true
	PYTHON_CMD="uv run python"
else
	if [ ! -d ".venv" ]; then
		echo "🧪 Creating local venv (.venv)"
		python -m venv .venv
	fi
	# shellcheck disable=SC1091
	source .venv/bin/activate
	python -m pip install -q -r requirements.txt || true
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Start backend in background (uvicorn inside app.py when __main__)
echo "🚀 Starting FastAPI Backend (Port 8080)..."
PORT=8080 $PYTHON_CMD app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background (production server)
FRONTEND_PORT="${PORT:-3000}"
echo "🎨 Starting Next.js Frontend (Port $FRONTEND_PORT)..."
cd youtube-summarizer-ui \
	&& BACKEND_URL="$BACKEND_URL" npm install --no-audit --no-fund \
	&& npm run build \
	&& BACKEND_URL="$BACKEND_URL" npm run start -- -p "$FRONTEND_PORT" &
FRONTEND_PID=$!

# Wait for both processes
echo ""
echo "✅ Both services started!"
echo "📱 Frontend: http://localhost:$FRONTEND_PORT"
echo "🔧 Backend API: http://localhost:8080"
echo "📊 Backend Health: http://localhost:8080/test"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for any process to finish
wait