#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    python3 \
    python3-pip \
    python3-venv

python3 -m pip install --user numpy pandas matplotlib

echo "Environment setup finished."
