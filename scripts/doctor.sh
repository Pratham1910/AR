#!/usr/bin/env bash
# Checks every part TVASTA needs on Linux and prints the fix for what's wrong.
#     bash scripts/doctor.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${TVASTA_VENV:-$ROOT/backend/.venv}"
problems=0
good() { printf '  [ok]   %s\n' "$1"; }
bad() { printf '  [FAIL] %s\n         fix: %s\n' "$1" "$2"; problems=$((problems + 1)); }
warn() { printf '  [warn] %s\n         %s\n' "$1" "$2"; }
get() { curl -fsS --max-time 4 "$1" 2>/dev/null; }

echo "TVASTA doctor"; echo
[ -x "$VENV/bin/python" ] && good "backend Python environment" || bad "backend environment missing" "bash scripts/setup_linux.sh"
[ -d "$ROOT/frontend/node_modules" ] && good "frontend packages" || bad "frontend/node_modules missing" "bash scripts/setup_linux.sh"
[ -f "$ROOT/models/mediapipe/hand_landmarker.task" ] && good "hand landmark model" || warn "hand model missing (hands via YOLO only)" "bash scripts/setup_linux.sh"

if docker info >/dev/null 2>&1; then
  docker exec tvasta-postgres pg_isready -U tvasta >/dev/null 2>&1 && good "PostgreSQL" || bad "PostgreSQL not ready" "docker compose up -d postgres"
  get http://127.0.0.1:9000/minio/health/live >/dev/null && good "MinIO" || bad "MinIO not reachable on 9000" "docker compose up -d minio"
else
  bad "Docker not usable" "sudo systemctl start docker; sudo usermod -aG docker \$USER (log out/in)"
fi

if status=$(get http://127.0.0.1:8000/api/system/status); then
  good "backend on 127.0.0.1:8000"
  echo "$status" | grep -q '"approximate":true' && warn "camera not calibrated" "AR view -> Calibrate camera (checkerboard), once per camera"
else
  owner=$(command -v ss >/dev/null && ss -ltnp 2>/dev/null | awk '$4 ~ /:8000$/ {print $6}')
  [ -n "$owner" ] && bad "port 8000 is taken by something else: $owner" "stop it, then bash scripts/start_linux.sh" \
                  || bad "backend not running" "bash scripts/start_linux.sh (log: .run/backend.log)"
fi

if pose=$(get http://127.0.0.1:8765/health); then
  echo "$pose" | grep -q '"cuda":true' && good "pose service on 8765 (GPU)" || bad "pose service up without a usable GPU" "check the NVIDIA driver (nvidia-smi), then restart the pose service"
else
  bad "pose service not reachable on 8765" "bash scripts/start_linux.sh (first time: bash pose_service/setup_wsl.sh); log: .run/pose.log"
fi

get http://127.0.0.1:5173 >/dev/null && good "frontend on 127.0.0.1:5173" || bad "frontend not running" "bash scripts/start_linux.sh (log: .run/frontend.log)"

if command -v nvidia-smi >/dev/null && gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
  good "GPU: $gpu"
else
  bad "no NVIDIA GPU / driver" "Ubuntu: sudo ubuntu-drivers install ; Arch/Omarchy: install the nvidia driver package, reboot"
fi

echo
[ "$problems" = 0 ] && echo "Everything looks fine." || echo "$problems problem(s) - see the fixes above."
