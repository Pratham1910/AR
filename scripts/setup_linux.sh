#!/usr/bin/env bash
# TVASTA one-time setup on Linux (Ubuntu / Debian, Arch / Omarchy, Fedora):
# backend, frontend, database. The GPU pose service runs natively too:
#     bash pose_service/setup_wsl.sh        (the same script works on plain Linux)
#
#     bash scripts/setup_linux.sh [--skip-docker] [--skip-frontend] [--no-system-packages]
#
# Python: the backend needs exactly 3.11; uv provides it whatever the system
# Python is (Ubuntu 24.04 has 3.12, Arch/Omarchy 3.13). Safe to run again.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
VENV="${TVASTA_VENV:-$BACKEND/.venv}"
SKIP_DOCKER=0; SKIP_FRONTEND=0; SYSTEM_PACKAGES=1
for arg in "$@"; do
  case "$arg" in
    --skip-docker) SKIP_DOCKER=1 ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --no-system-packages) SYSTEM_PACKAGES=0 ;;   # no sudo: they are installed already
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
ok() { printf '    ok: %s\n' "$1"; }
fail() { printf '\nSETUP STOPPED: %s\n' "$1" >&2; exit 1; }

if grep -qi microsoft /proc/version 2>/dev/null && [ "$VENV" = "$BACKEND/.venv" ] && [[ "$ROOT" == /mnt/* ]]; then
  fail "This is WSL on a Windows checkout: the backend runs on Windows there (scripts\\setup.ps1). Set TVASTA_VENV to try it here anyway."
fi

# ── 1. System packages ───────────────────────────────────────────────────
step "System packages"
if [ "$SYSTEM_PACKAGES" = 0 ]; then
  echo "    skipped (--no-system-packages)"
elif command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git curl unzip build-essential psmisc libgl1 libglib2.0-0 libegl1 >/dev/null
elif command -v pacman >/dev/null; then          # Arch, Omarchy, Manjaro
  sudo pacman -S --needed --noconfirm git curl unzip base-devel psmisc mesa libglvnd glib2 >/dev/null
elif command -v dnf >/dev/null; then             # Fedora
  sudo dnf install -y -q git curl unzip gcc gcc-c++ make psmisc mesa-libGL mesa-libEGL glib2
else
  echo "    unknown package manager: make sure git, curl, unzip, a C compiler, libGL/libEGL and glib are installed"
fi
ok "system packages"

# ── 2. Tools ─────────────────────────────────────────────────────────────
step "uv (provides Python 3.11)"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

if [ "$SKIP_FRONTEND" = 0 ]; then
  if ! command -v node >/dev/null || [ "$(node -v | cut -c2- | cut -d. -f1)" -lt 18 ]; then
    fail "Node.js 18+ needed. Ubuntu: https://github.com/nodesource/distributions ; Arch/Omarchy: sudo pacman -S nodejs npm"
  fi
  ok "Node.js $(node -v)"
fi

# ── 3. Database + object storage ─────────────────────────────────────────
if [ "$SKIP_DOCKER" = 0 ]; then
  step "PostgreSQL + MinIO (docker compose)"
  command -v docker >/dev/null || fail "Docker not found. Ubuntu: https://docs.docker.com/engine/install/ubuntu/ ; Arch/Omarchy: sudo pacman -S docker docker-compose && sudo systemctl enable --now docker"
  docker info >/dev/null 2>&1 || fail "Docker isn't usable by $USER: sudo systemctl start docker; sudo usermod -aG docker $USER (then log out and in)"
  (cd "$ROOT" && docker compose up -d postgres minio)
  for _ in $(seq 60); do docker exec tvasta-postgres pg_isready -U tvasta >/dev/null 2>&1 && break; sleep 2; done
  docker exec tvasta-postgres pg_isready -U tvasta >/dev/null 2>&1 || fail "PostgreSQL didn't become ready (docker logs tvasta-postgres)"
  ok "PostgreSQL :5432, MinIO :9000"
fi

# ── 4. Backend Python environment ────────────────────────────────────────
step "Backend: Python 3.11 environment ($VENV)"
[ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
PY="$VENV/bin/python"
UV_HTTP_TIMEOUT=900 uv pip install --python "$PY" -r "$BACKEND/requirements.txt"
# Hand masks (MediaPipe) without its declared deps — they add a second OpenCV.
if ! uv pip install --python "$PY" --no-deps "mediapipe==1.1.0" "absl-py~=2.3" "flatbuffers~=25.9" "sounddevice~=0.5"; then
  echo "    warning: MediaPipe not installed — hands are then found by YOLO's person class only"
fi
# Exactly one OpenCV (the contrib build: it has ArUco).
uv pip uninstall --python "$PY" opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
uv pip install --python "$PY" --reinstall --no-deps "opencv-contrib-python-headless==4.10.0.84"
"$PY" -c "import cv2, sys; sys.exit(0 if hasattr(cv2.aruco, 'ArucoDetector') else 1)" || fail "OpenCV check failed (cv2.aruco missing)"
ok "Python packages, OpenCV contrib"

mkdir -p "$ROOT/models/mediapipe"
[ -f "$ROOT/models/mediapipe/hand_landmarker.task" ] || curl -sSL -o "$ROOT/models/mediapipe/hand_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
[ -f "$BACKEND/yolov8n-seg.pt" ] || curl -sSL -o "$BACKEND/yolov8n-seg.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
ok "hand + YOLO models"

# ── 5. Database schema ───────────────────────────────────────────────────
if [ "$SKIP_DOCKER" = 0 ]; then
  step "Database schema"
  (cd "$BACKEND" && "$PY" -m alembic upgrade head) || fail "migration failed — is PostgreSQL on localhost:5432 (user/password tvasta)?"
  ok "schema up to date"
fi

# ── 6. Frontend ──────────────────────────────────────────────────────────
if [ "$SKIP_FRONTEND" = 0 ]; then
  step "Frontend packages"
  (cd "$ROOT/frontend" && npm ci --no-audit --no-fund) || fail "npm ci failed"
  ok "frontend ready"
fi

step "Self-test"
(cd "$BACKEND" && "$PY" -c "import app.main; print('backend imports OK')") || fail "the backend does not import (see above)"

printf '\nDone. Next:\n  bash pose_service/setup_wsl.sh   # GPU pose service (works on plain Linux too)\n  bash scripts/start_linux.sh      # every day\n  bash scripts/doctor.sh           # check everything\n'
