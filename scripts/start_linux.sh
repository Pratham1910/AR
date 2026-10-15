#!/usr/bin/env bash
# Starts TVASTA on Linux in the background (logs in .run/*.log):
# PostgreSQL + MinIO (docker), the GPU pose service, the backend, the frontend.
#     bash scripts/start_linux.sh [--no-pose]      stop with: bash scripts/stop_linux.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/.run"; mkdir -p "$RUN"
VENV="${TVASTA_VENV:-$ROOT/backend/.venv}"
NO_POSE=0; [ "${1:-}" = "--no-pose" ] && NO_POSE=1

busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
launch() {  # name, workdir, command...
  local name=$1 dir=$2; shift 2
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then echo "    $name already running"; return; fi
  (cd "$dir" && setsid nohup "$@" >"$RUN/$name.log" 2>&1 & echo $! >"$RUN/$name.pid")
  echo "    $name started (log: .run/$name.log)"
}

[ -x "$VENV/bin/python" ] || { echo "Run scripts/setup_linux.sh first."; exit 1; }
echo "==> Database + storage"; (cd "$ROOT" && docker compose up -d postgres minio >/dev/null)
if [ "$NO_POSE" = 0 ]; then
  echo "==> Pose service :8765"
  if busy 8765; then echo "    already running"; else launch pose "$ROOT" bash pose_service/start.sh; fi
fi
echo "==> Backend :8000"
if busy 8000; then echo "    port 8000 already in use (bash scripts/doctor.sh shows by what)"; else
  launch backend "$ROOT/backend" "$VENV/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000; fi
echo "==> Frontend :5173"
if busy 5173; then echo "    already running"; else launch frontend "$ROOT/frontend" npm run dev -- --host 127.0.0.1; fi

for _ in $(seq 60); do busy 5173 && break; sleep 1; done
command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:5173 >/dev/null 2>&1 || true
echo "Started: http://127.0.0.1:5173  (the pose service needs ~1 min to load). Check: bash scripts/doctor.sh"
