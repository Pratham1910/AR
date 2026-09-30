#!/usr/bin/env bash
# Starts the MegaPose pose service inside WSL. Run from WSL:
#   bash /mnt/c/Users/PRATHAMESH/Desktop/Toolkit/AR/TVASTA/pose_service/start.sh
set -euo pipefail

HAPPYPOSE_DIR="${HAPPYPOSE_DIR:-$HOME/tvasta-pose/happypose}"
export HAPPYPOSE_DATA_DIR="${HAPPYPOSE_DATA_DIR:-$HOME/tvasta-pose/data}"
# HappyPose's Panda3D renderer (used by MegaPose itself) asserts this is set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$HAPPYPOSE_DIR/.venv/bin/python" -m uvicorn server:app \
  --app-dir "$SERVICE_DIR" --host 0.0.0.0 --port "${PORT:-8765}"
