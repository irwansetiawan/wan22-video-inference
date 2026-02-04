#!/bin/bash
set -e

echo "=== WAN 2.2 EC2 Setup ==="

# Configuration
DATA_DIR="${DATA_DIR:-/data}"
MODEL_NAME="Wan-AI/Wan2.2-TI2V-5B"

# Create data directories
echo "Creating directories..."
sudo mkdir -p "$DATA_DIR"
sudo chown -R ubuntu:ubuntu "$DATA_DIR"
mkdir -p "$DATA_DIR/inputs" "$DATA_DIR/outputs"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Create virtual environment
echo "Creating virtual environment..."
cd "$DATA_DIR"
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "huggingface_hub[cli]"

# Clone WAN 2.2 repository
echo "Cloning WAN 2.2 repository..."
if [ ! -d "$DATA_DIR/Wan2.2" ]; then
    git clone https://github.com/Wan-Video/Wan2.2.git "$DATA_DIR/Wan2.2"
fi

# Install WAN 2.2 dependencies
echo "Installing WAN 2.2 dependencies..."
cd "$DATA_DIR/Wan2.2"
pip install -r requirements.txt

# Download model weights (this takes a while)
echo "Downloading model weights (~20GB)..."
huggingface-cli download "$MODEL_NAME" --local-dir "$DATA_DIR/Wan2.2-TI2V-5B"

# Install API dependencies
echo "Installing API dependencies..."
pip install fastapi uvicorn python-dotenv boto3 httpx pydantic

# Setup systemd service
echo "Setting up systemd service..."
sudo tee /etc/systemd/system/wan-api.service > /dev/null <<EOF
[Unit]
Description=WAN 2.2 Video Generation API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/wan22
Environment=PATH=$DATA_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=DATA_DIR=$DATA_DIR
EnvironmentFile=/home/ubuntu/wan22/.env
ExecStart=$DATA_DIR/venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable wan-api

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your project to /home/ubuntu/wan22"
echo "2. Create .env file with required variables"
echo "3. Run: sudo systemctl start wan-api"
echo "4. Check logs: sudo journalctl -u wan-api -f"
