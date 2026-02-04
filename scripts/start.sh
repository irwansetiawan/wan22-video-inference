#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"

# Activate virtual environment
source "$DATA_DIR/venv/bin/activate"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

export DATA_DIR

echo "Starting WAN 2.2 API server..."
exec uvicorn src.server:app --host 0.0.0.0 --port 8000
