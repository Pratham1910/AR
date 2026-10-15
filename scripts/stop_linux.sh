#!/usr/bin/env bash
# Stops what scripts/start_linux.sh started (the database containers keep running;
# `docker compose stop` in the repo folder stops those too).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in frontend backend pose; do
  pid_file="$ROOT/.run/$name.pid"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true   # its whole process group
    rm -f "$pid_file"
    echo "stopped $name"
  fi
done
command -v fuser >/dev/null && fuser -k 8765/tcp >/dev/null 2>&1 || true   # MegaPose renderer workers
