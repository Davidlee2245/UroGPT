#!/bin/bash

# Start UroGPT FastAPI Backend
# This script starts the API server on port 8000

echo "🚀 Starting UroGPT API Server..."
echo "================================"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate urogpt

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set!"
    echo "Set it with: export OPENAI_API_KEY='your-key-here'"
    echo ""
fi

# Kill existing process on port 8000
PORT_PID=$(lsof -ti:8000)
if [ ! -z "$PORT_PID" ]; then
    echo "🔄 Killing existing process on port 8000 (PID: $PORT_PID)"
    kill -9 $PORT_PID
    sleep 1
fi

# Start API server
cd /home/david/.cursor-tutor/UroGPT
echo "📡 Starting API server on http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo ""
python api/app.py

