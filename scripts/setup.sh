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
echo "Downloading WAN 2.2 14B FP8 models (~34GB total)..."
pip install "huggingface_hub[cli]"

MODELS_DIR="$COMFYUI_DIR/models"

# Text encoder (FP8, ~5GB)
echo "Downloading text encoder..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# VAE (~254MB)
echo "Downloading VAE..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/vae/wan_2.1_vae.safetensors" \
    "$MODELS_DIR/vae/wan_2.1_vae.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/vae/wan_2.1_vae.safetensors" \
    "$MODELS_DIR/vae/wan_2.1_vae.safetensors"

# T2V diffusion models (FP8, ~14GB each)
echo "Downloading T2V high-noise model..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"

echo "Downloading T2V low-noise model..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"

# I2V diffusion models (FP8, ~14GB each)
echo "Downloading I2V high-noise model..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

echo "Downloading I2V low-noise model..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "$MODELS_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

# CLIP Vision for I2V (~3.9GB)
echo "Downloading CLIP Vision encoder..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/clip_vision/clip_vision_h.safetensors \
    --local-dir "$MODELS_DIR"
ln -sf "$MODELS_DIR/split_files/clip_vision/clip_vision_h.safetensors" \
    "$MODELS_DIR/clip_vision/clip_vision_h.safetensors" 2>/dev/null || \
    cp "$MODELS_DIR/split_files/clip_vision/clip_vision_h.safetensors" \
    "$MODELS_DIR/clip_vision/clip_vision_h.safetensors"

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
