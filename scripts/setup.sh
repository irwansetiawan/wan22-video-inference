#!/bin/bash
set -e

echo "=== WAN 2.2 ComfyUI Setup ==="

# Configuration
DATA_DIR="${DATA_DIR:-/data}"
COMFYUI_DIR="$DATA_DIR/ComfyUI"

# Create data directories
echo "Creating directories..."
sudo mkdir -p "$DATA_DIR"
sudo chown -R ubuntu:ubuntu "$DATA_DIR"
mkdir -p "$DATA_DIR/inputs" "$DATA_DIR/outputs"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# =============================================
# ComfyUI Installation
# =============================================
echo "Installing ComfyUI..."
if [ ! -d "$COMFYUI_DIR" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

cd "$COMFYUI_DIR"
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI dependencies
echo "Installing ComfyUI dependencies..."
pip install -r requirements.txt

# =============================================
# Download WAN 2.2 14B FP8 Models
# =============================================
echo "Downloading WAN 2.2 14B FP8 models (~60GB total)..."
pip install "huggingface_hub[cli]"

MODELS_DIR="$COMFYUI_DIR/models"
HF_DOWNLOAD="$COMFYUI_DIR/venv/bin/python -c"

# Download helper: downloads from HuggingFace and symlinks into ComfyUI model dirs
download_model() {
    local repo="$1"
    local remote_path="$2"
    local target_dir="$3"
    local filename=$(basename "$remote_path")
    local target="$MODELS_DIR/$target_dir/$filename"

    if [ -f "$target" ] || [ -L "$target" ]; then
        echo "  SKIP $filename (already exists)"
        return
    fi

    echo "  Downloading $filename..."
    local local_path=$($HF_DOWNLOAD "
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='$repo', filename='$remote_path'))
")
    mkdir -p "$MODELS_DIR/$target_dir"
    ln -sf "$local_path" "$target"
    echo "  -> $target"
}

# Text encoder (FP8, ~5GB)
echo "Downloading text encoder..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "text_encoders"

# VAE (~254MB)
echo "Downloading VAE..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/vae/wan_2.1_vae.safetensors" "vae"

# T2V diffusion models (FP8, ~14GB each)
echo "Downloading T2V high-noise model..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" "diffusion_models"

echo "Downloading T2V low-noise model..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" "diffusion_models"

# I2V diffusion models (FP8, ~14GB each)
echo "Downloading I2V high-noise model..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" "diffusion_models"

echo "Downloading I2V low-noise model..."
download_model "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" "diffusion_models"

# CLIP Vision for I2V (~1.3GB, from WAN 2.1 repo — shared across versions)
echo "Downloading CLIP Vision encoder..."
download_model "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
    "split_files/clip_vision/clip_vision_h.safetensors" "clip_vision"

# =============================================
# API Server Dependencies
# =============================================
echo "Installing API dependencies..."
cd /home/ubuntu/wan22
python3 -m venv /data/venv
source /data/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# =============================================
# Add swap space (16GB RAM is tight)
# =============================================
if [ ! -f /swapfile ]; then
    echo "Adding 16GB swap..."
    sudo fallocate -l 16G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# =============================================
# Systemd Services
# =============================================
echo "Setting up systemd services..."

# ComfyUI service
sudo tee /etc/systemd/system/comfyui.service > /dev/null <<EOF
[Unit]
Description=ComfyUI Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$COMFYUI_DIR
ExecStart=$COMFYUI_DIR/venv/bin/python main.py --listen 127.0.0.1 --port 8188
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# WAN API service
sudo tee /etc/systemd/system/wan-api.service > /dev/null <<EOF
[Unit]
Description=WAN 2.2 Video Generation API
After=network.target comfyui.service
Requires=comfyui.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/wan22
Environment=PATH=/data/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=DATA_DIR=$DATA_DIR
EnvironmentFile=/home/ubuntu/wan22/.env
ExecStart=/data/venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable comfyui wan-api

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your project to /home/ubuntu/wan22"
echo "2. Create .env file with required variables"
echo "3. Start services:"
echo "   sudo systemctl start comfyui"
echo "   sudo systemctl start wan-api"
echo "4. Check logs:"
echo "   sudo journalctl -u comfyui -f"
echo "   sudo journalctl -u wan-api -f"
