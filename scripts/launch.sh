#!/bin/bash
# =============================================================================
# Launch Script for 5D Regressor
# =============================================================================
# This script launches the entire technology stack locally.
#
# Usage:
#   ./scripts/launch.sh          # Launch both backend and frontend
#   ./scripts/launch.sh backend  # Launch only backend
#   ./scripts/launch.sh frontend # Launch only frontend
#   ./scripts/launch.sh docker   # Launch using Docker
#
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================"
echo "5D Regressor - Launch Script"
echo -e "========================================${NC}"

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        return 1
    fi
    return 0
}

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to launch backend
launch_backend() {
    echo ""
    echo -e "${GREEN}Starting Backend Server...${NC}"
    echo "----------------------------------------"
    
    cd "$PROJECT_ROOT/backend"
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Install package if needed
    pip install -e . --quiet 2>/dev/null || true
    
    # Check port
    if ! check_port 8000; then
        echo "Attempting to free port 8000..."
        kill $(lsof -t -i:8000) 2>/dev/null || true
        sleep 1
    fi
    
    # Launch uvicorn
    echo "Backend starting on http://localhost:8000"
    uvicorn fivedreg.main:app --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
    
    # Wait for backend to be ready
    echo "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}Backend is ready!${NC}"
            break
        fi
        sleep 1
    done
}

# Function to launch frontend
launch_frontend() {
    echo ""
    echo -e "${GREEN}Starting Frontend Server...${NC}"
    echo "----------------------------------------"
    
    cd "$PROJECT_ROOT/frontend"
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install --silent
    fi
    
    # Check port
    if ! check_port 3000; then
        echo "Attempting to free port 3000..."
        kill $(lsof -t -i:3000) 2>/dev/null || true
        sleep 1
    fi
    
    # Launch Next.js
    echo "Frontend starting on http://localhost:3000"
    npm run dev &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
}

# Function to launch with Docker
launch_docker() {
    echo ""
    echo -e "${GREEN}Starting with Docker Compose...${NC}"
    echo "----------------------------------------"
    
    cd "$PROJECT_ROOT"
    
    # Check Docker
    if ! check_command docker; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Launch
    docker-compose up --build -d
    
    echo ""
    echo -e "${GREEN}Services started!${NC}"
    echo "Frontend: http://localhost:3000"
    echo "Backend:  http://localhost:8000"
    echo "API Docs: http://localhost:8000/docs"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
}

# Function to open browser
open_browser() {
    sleep 3
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:3000"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open "http://localhost:3000"
        fi
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    echo "Done."
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Main logic
case "${1:-all}" in
    backend)
        check_command python || exit 1
        check_command pip || exit 1
        launch_backend
        wait
        ;;
    frontend)
        check_command node || exit 1
        check_command npm || exit 1
        launch_frontend
        wait
        ;;
    docker)
        launch_docker
        ;;
    all|*)
        check_command python || exit 1
        check_command pip || exit 1
        check_command node || exit 1
        check_command npm || exit 1
        
        launch_backend
        launch_frontend
        
        echo ""
        echo -e "${GREEN}========================================"
        echo "All services started!"
        echo "========================================${NC}"
        echo ""
        echo "Frontend: http://localhost:3000"
        echo "Backend:  http://localhost:8000"
        echo "API Docs: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop all services"
        echo ""
        
        open_browser
        
        # Wait for both processes
        wait
        ;;
esac