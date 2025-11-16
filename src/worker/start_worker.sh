#!/bin/bash
#
# NANO-OS Celery Worker Startup Script
# =====================================
#
# This script starts the Celery worker for NANO-OS simulations.
#
# Usage:
#   ./start_worker.sh                    # Start with defaults
#   ./start_worker.sh --dev              # Development mode (autoreload)
#   ./start_worker.sh --production       # Production mode (4 workers)
#   ./start_worker.sh --debug            # Debug mode (verbose logging)
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
LOGLEVEL="info"
CONCURRENCY=2
QUEUES="simulations,default"
AUTORELOAD=""
MAX_TASKS_PER_CHILD=100
POOL="prefork"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Print header
print_header() {
    echo ""
    print_msg "$BLUE" "╔══════════════════════════════════════════════════════════╗"
    print_msg "$BLUE" "║         NANO-OS Celery Worker Startup Script           ║"
    print_msg "$BLUE" "╚══════════════════════════════════════════════════════════╝"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_msg "$YELLOW" "Checking prerequisites..."

    # Check if we're in the project root
    if [ ! -f "requirements.txt" ]; then
        print_msg "$RED" "Error: Must run from project root"
        print_msg "$RED" "Current directory: $(pwd)"
        exit 1
    fi

    # Check if Redis is running
    if ! redis-cli ping > /dev/null 2>&1; then
        print_msg "$RED" "Error: Redis is not running"
        print_msg "$YELLOW" "Start Redis with: docker run -d --name redis -p 6379:6379 redis:7-alpine"
        exit 1
    fi

    print_msg "$GREEN" "✓ Redis is running"

    # Check if celery is installed
    if ! python -c "import celery" 2>/dev/null; then
        print_msg "$RED" "Error: Celery is not installed"
        print_msg "$YELLOW" "Install with: pip install -r requirements.txt"
        exit 1
    fi

    print_msg "$GREEN" "✓ Celery is installed"

    # Check if worker module exists
    if [ ! -d "src/worker" ]; then
        print_msg "$RED" "Error: src/worker directory not found"
        exit 1
    fi

    print_msg "$GREEN" "✓ Worker module found"
    echo ""
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|--development)
                LOGLEVEL="debug"
                CONCURRENCY=1
                AUTORELOAD="--autoreload"
                POOL="solo"
                print_msg "$YELLOW" "Mode: Development (autoreload enabled)"
                shift
                ;;
            --prod|--production)
                LOGLEVEL="info"
                CONCURRENCY=4
                MAX_TASKS_PER_CHILD=100
                print_msg "$YELLOW" "Mode: Production (4 workers)"
                shift
                ;;
            --debug)
                LOGLEVEL="debug"
                print_msg "$YELLOW" "Mode: Debug (verbose logging)"
                shift
                ;;
            --solo)
                POOL="solo"
                CONCURRENCY=1
                print_msg "$YELLOW" "Pool: Solo (single process)"
                shift
                ;;
            --gevent)
                POOL="gevent"
                CONCURRENCY=100
                print_msg "$YELLOW" "Pool: Gevent (async, 100 workers)"
                shift
                ;;
            -c|--concurrency)
                CONCURRENCY=$2
                shift 2
                ;;
            -Q|--queues)
                QUEUES=$2
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dev, --development    Development mode (autoreload, debug)"
                echo "  --prod, --production    Production mode (4 workers)"
                echo "  --debug                 Debug logging"
                echo "  --solo                  Solo pool (single process)"
                echo "  --gevent                Gevent pool (async, 100 workers)"
                echo "  -c, --concurrency N     Number of worker processes"
                echo "  -Q, --queues QUEUES     Comma-separated queue names"
                echo "  -h, --help              Show this help"
                echo ""
                exit 0
                ;;
            *)
                print_msg "$RED" "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Start the worker
start_worker() {
    print_msg "$GREEN" "Starting Celery worker..."
    echo ""
    print_msg "$BLUE" "Configuration:"
    print_msg "$BLUE" "  Log Level:     $LOGLEVEL"
    print_msg "$BLUE" "  Concurrency:   $CONCURRENCY"
    print_msg "$BLUE" "  Pool:          $POOL"
    print_msg "$BLUE" "  Queues:        $QUEUES"
    print_msg "$BLUE" "  Max Tasks:     $MAX_TASKS_PER_CHILD"
    if [ -n "$AUTORELOAD" ]; then
        print_msg "$BLUE" "  Autoreload:    enabled"
    fi
    echo ""
    print_msg "$YELLOW" "Press Ctrl+C to stop the worker"
    echo ""

    # Build command
    CMD="celery -A src.worker.celery_app worker"
    CMD="$CMD --loglevel=$LOGLEVEL"
    CMD="$CMD -Q $QUEUES"
    CMD="$CMD --pool=$POOL"

    if [ "$POOL" != "solo" ]; then
        CMD="$CMD -c $CONCURRENCY"
        CMD="$CMD --max-tasks-per-child=$MAX_TASKS_PER_CHILD"
    fi

    if [ -n "$AUTORELOAD" ]; then
        CMD="$CMD $AUTORELOAD"
    fi

    # Print command
    print_msg "$BLUE" "Command: $CMD"
    echo ""

    # Execute
    exec $CMD
}

# Main
main() {
    print_header
    parse_args "$@"
    check_prerequisites
    start_worker
}

main "$@"
