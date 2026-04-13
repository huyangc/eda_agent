#!/bin/bash
# =============================================================================
# EDA Agent & RAG Service Restart Script
# =============================================================================

# Stop on first error instead of proceeding
set -e

# Setup variables
PYTHON_EXEC="/opt/homebrew/Caskroom/miniconda/base/bin/python"
EDA_DIR="/Users/hzf/workspace/eda_agent"

echo "🔄 [1/4] Checking for running services on ports 8000 and 9000..."
PIDS=$(lsof -t -i:8000,9000 2>/dev/null || true)

if [ -n "$PIDS" ]; then
    echo "🛑 [2/4] Killing old processes: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 1
else
    echo "✨ [2/4] No existing processes found on ports 8000 or 9000."
fi

# Move to working directory
cd "$EDA_DIR"

echo "🚀 [3/4] Starting RAG Service (Port 9000)..."
nohup "$PYTHON_EXEC" rag_service/main.py > rag_service.log 2>&1 &
RAG_PID=$!
echo "   -> RAG Service started with PID $RAG_PID (logging to rag_service.log)"

echo "🚀 [4/4] Starting EDA Agent (Port 8000)..."
nohup "$PYTHON_EXEC" app/main.py > eda_agent_startup.log 2>&1 &
EDA_PID=$!
echo "   -> EDA Agent started with PID $EDA_PID (logging to eda_agent_startup.log)"

echo ""
echo "✅ Restart complete! Services are running in the background."
echo "   - RAG Service:  http://localhost:9000"
echo "   - EDA Agent:    http://localhost:8000"
echo ""
echo "💡 To view logs:"
echo "   tail -f rag_service.log"
echo "   tail -f eda_agent_startup.log"
echo "   tail -f logs/YYYY-MM-DD/*.jsonl"
