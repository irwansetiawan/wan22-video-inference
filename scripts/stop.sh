#!/bin/bash

echo "Stopping WAN 2.2 API server..."

# Find and kill uvicorn process
pkill -f "uvicorn src.server:app" || true

echo "Server stopped"
