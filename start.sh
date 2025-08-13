#!/bin/bash

# YouTube Summarizer Development Startup Script
echo "🎬 Starting YouTube Summarizer Development Environment"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8080"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start backend in background
echo "🚀 Starting FastAPI Backend (Port 8080)..."
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "🎨 Starting Next.js Frontend (Port 3000)..."
cd youtube-summarizer-ui && npm run dev &
FRONTEND_PID=$!

# Wait for both processes
echo ""
echo "✅ Both services started!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8080"
echo "📊 Backend Health: http://localhost:8080/test"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for any process to finish
wait