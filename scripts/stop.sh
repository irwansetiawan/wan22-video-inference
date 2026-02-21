#!/bin/bash

echo "Stopping WAN 2.2 services..."

sudo systemctl stop wan-api 2>/dev/null || true
echo "API server stopped"

sudo systemctl stop comfyui 2>/dev/null || true
echo "ComfyUI stopped"

echo "All services stopped"
