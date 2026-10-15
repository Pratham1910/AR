#!/usr/bin/env bash
# One-time setup of the GPU pose service (MegaPose + FastSAM + DINOv2): inside
# WSL 2 on Windows, or natively on Linux (Ubuntu/Debian, Arch/Omarchy, Fedora).
# Run from the repository folder:
#
#     bash pose_service/setup_wsl.sh
#
# Needs: an NVIDIA GPU with a working driver — on WSL the WINDOWS driver (WSL
# gets CUDA from it; don't install a Linux driver inside WSL), on Linux the
# distribution's NVIDIA driver. No CUDA toolkit needed (PyTorch brings its
# own CUDA 12.4 runtime). ~12 GB free disk, internet.
# Safe to run again: finished steps are skipped. Takes 15-40 min the first time.
set -euo pipefail

POSE_HOME="${TVASTA_POSE_HOME:-$HOME/tvasta-pose}"
HAPPYPOSE_DIR="$POSE_HOME/happypose"
# The HappyPose commit the reference PC runs (its `dev` branch moves).
HAPPYPOSE_COMMIT="6278e12b9ada314a5def8041d9f33c649fe82f40"
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nSETUP STOPPED: %s\n' "$1" >&2; exit 1; }

step "GPU visible"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  if grep -qi microsoft /proc/version 2>/dev/null; then
    fail "nvidia-smi doesn't work in WSL. Update the NVIDIA driver on WINDOWS, run 'wsl --shutdown' in PowerShell, reopen WSL and try again."
  fi
  fail "nvidia-smi doesn't work. Install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install; Arch/Omarchy: the nvidia / nvidia-open package) and reboot."
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

step "System packages"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git curl unzip build-essential psmisc libgl1 libglib2.0-0 libegl1 >/dev/null
elif command -v pacman >/dev/null; then          # Arch, Omarchy, Manjaro
  sudo pacman -S --needed --noconfirm git curl unzip base-devel psmisc mesa libglvnd glib2 >/dev/null
elif command -v dnf >/dev/null; then             # Fedora
  sudo dnf install -y -q git curl unzip gcc gcc-c++ make psmisc mesa-libGL mesa-libEGL glib2
else
  echo "    unknown package manager: make sure git, curl, unzip, a C compiler, libGL/libEGL and glib are installed"
fi

step "uv (Python package manager)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "HappyPose source (pinned commit)"
mkdir -p "$POSE_HOME"
if [ ! -d "$HAPPYPOSE_DIR/.git" ]; then
  git clone --recurse-submodules https://github.com/agimus-project/happypose.git "$HAPPYPOSE_DIR"
fi
git -C "$HAPPYPOSE_DIR" fetch --quiet origin
git -C "$HAPPYPOSE_DIR" checkout --quiet "$HAPPYPOSE_COMMIT"
git -C "$HAPPYPOSE_DIR" submodule update --init --recursive --quiet

step "Python 3.10 environment with the pinned packages (CUDA 12.4 PyTorch)"
cd "$HAPPYPOSE_DIR"
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
echo "setuptools<70" > "$POSE_HOME/build-constraints.txt"   # visdom still needs pkg_resources to build
UV_HTTP_TIMEOUT=900 uv pip install --python .venv/bin/python \
  -r "$SERVICE_DIR/requirements-lock.txt" \
  --build-constraints "$POSE_HOME/build-constraints.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
# HappyPose itself, editable from the clone (its dependencies are in the lock file).
uv pip install --python .venv/bin/python --no-deps -e . -e happypose/pose_estimators/cosypose

step "MegaPose model weights"
export HAPPYPOSE_DATA_DIR="$POSE_HOME/data"
if [ ! -d "$HAPPYPOSE_DATA_DIR/megapose-models/coarse-rgb-906902141" ]; then
  .venv/bin/python -m happypose.toolbox.utils.download --megapose_models
fi

step "FastSAM + DINOv2 weights (finding objects by their 3D model)"
mkdir -p "$POSE_HOME/weights" && cd "$POSE_HOME/weights"
[ -f FastSAM-s.pt ] || curl -sSLO https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt
[ -f dinov2_vits14_pretrain.pth ] || curl -sSO https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
if [ ! -d dinov2 ]; then
  curl -sSL -o dinov2.zip https://github.com/facebookresearch/dinov2/zipball/main
  unzip -q dinov2.zip && mv facebookresearch-dinov2-* dinov2 && rm dinov2.zip
fi

step "Self-test"
cd "$HAPPYPOSE_DIR"
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))
import happypose, panda3d, fastapi, trimesh, ultralytics  # noqa: F401
print("happypose, panda3d, fastapi, trimesh, ultralytics import OK")
PY

printf '\nPose service ready. Start it with:\n    bash %s/start.sh\nand check:  curl http://127.0.0.1:8765/health   (expect "cuda": true)\n' "$SERVICE_DIR"
