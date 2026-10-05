#!/usr/bin/env bash

# ==============================================================================
# 🚀 1-Click Complete System Launcher (arXiv AI Research Assistant & Benchmark)
# ==============================================================================


#ACTIVATE VIRTUAL ENVIRONMENT  "source .venv/bin/activate "
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  🤖 Starting arXiv Research Assistant & Evaluation Platform   ${NC}"
echo -e "${CYAN}================================================================${NC}"

# 1. Check & Setup .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}ℹ .env not found. Creating from .env.example...${NC}"
    cp .env.example .env
fi

# 2. Virtual Environment Setup
if [ -d ".venv" ]; then
    source .venv/bin/activate
    export PATH="$PROJECT_DIR/.venv/bin:$PATH"
    export VIRTUAL_ENV="$PROJECT_DIR/.venv"
elif command -v uv >/dev/null 2>&1; then
    echo -e "${BLUE}▶ Creating virtual environment using uv...${NC}"
    uv sync
    source .venv/bin/activate
    export PATH="$PROJECT_DIR/.venv/bin:$PATH"
    export VIRTUAL_ENV="$PROJECT_DIR/.venv"
else
    echo -e "${BLUE}▶ Creating virtual environment using python3...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    export PATH="$PROJECT_DIR/.venv/bin:$PATH"
    export VIRTUAL_ENV="$PROJECT_DIR/.venv"
    pip install -r requirements.txt
fi

# 3. Check & Install Frontend Dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${BLUE}▶ Installing frontend dependencies (npm install)...${NC}"
    (cd frontend && npm install)
fi

# 4. Retrieval backends are cloud services configured in .env.
echo -e "\n${BLUE}[1/3] Using configured Qdrant Cloud and Elasticsearch Cloud services.${NC}"

# 5. Trap Ctrl+C for clean shutdown
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down backend and frontend servers...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    echo -e "${GREEN}✓ All services stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 6. Start FastAPI Backend
echo -e "\n${BLUE}[2/3] Starting FastAPI Backend (:8000)...${NC}"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# 7. Start Next.js Frontend
echo -e "${BLUE}[3/3] Starting Next.js Frontend (:3000)...${NC}"
(cd frontend && npm run dev) &
FRONTEND_PID=$!

# 8. Print Ready Banner
echo -e "\n${GREEN}================================================================${NC}"
echo -e "${BOLD}${GREEN}  ✨ SYSTEM IS LIVE AND READY! ✨${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "${CYAN} • Chat & Evaluation UI:${NC} ${BOLD}http://localhost:3000${NC}"
echo -e "${CYAN} • 2×2 Eval Benchmark:${NC}   ${BOLD}http://localhost:3000/evaluation${NC}"
echo -e "${CYAN} • Backend API Docs:${NC}     ${BOLD}http://localhost:8000/docs${NC}"
echo -e "${CYAN} • Health Endpoint:${NC}      ${BOLD}http://localhost:8000/api/health${NC}"
echo -e "${YELLOW}----------------------------------------------------------------${NC}"
echo -e "${YELLOW}Press Ctrl+C at any time to stop all servers.${NC}\n"

# Keep script running and wait for background jobs
wait "$BACKEND_PID" "$FRONTEND_PID"
