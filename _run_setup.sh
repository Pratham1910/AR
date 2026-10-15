#!/bin/bash
export TVASTA_VENV=/tmp/tvasta-venv-test
bash scripts/setup_linux.sh --skip-docker --skip-frontend --no-system-packages
echo "EXIT=$?"
