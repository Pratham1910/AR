# Model-based (CAD) pose service

MegaPose (via [HappyPose](https://github.com/agimus-project/happypose)) matches
a registered 3D model's own mesh against a webcam frame and returns its full
6DoF pose. It runs in WSL2 with CUDA; the Windows backend calls it at
`POSE_SERVICE_URL` (default `http://localhost:8765`) from
`POST /api/vision/model-pose`.

## One-time setup (WSL, Ubuntu 22.04)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/tvasta-pose && cd ~/tvasta-pose
git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
cd happypose
echo "setuptools<70" > ../build-constraints.txt   # visdom still needs pkg_resources to build
uv venv --python 3.10 .venv
UV_HTTP_TIMEOUT=900 uv pip install --python .venv/bin/python \
  -r requirements/pypi.txt -r requirements/cu124.txt \
  --build-constraints ../build-constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python fastapi uvicorn fast-simplification
HAPPYPOSE_DATA_DIR=~/tvasta-pose/data .venv/bin/python -m happypose.toolbox.utils.download --megapose_models
```

## Run

```bash
bash /mnt/c/Users/PRATHAMESH/Desktop/Toolkit/AR/TVASTA/pose_service/start.sh
curl http://localhost:8765/health   # expect "cuda": true
```

Then, in the web app: 3D Viewer → AR Registration → **Model-based CAD**.

## How it works / limits

- The backend uploads each Model3D's GLB once (scaled by `Model3D.scale`,
  recentered on its bounding box — the same frame the frontend renders in).
  Re-registered automatically if the scale changes.
- First frame: YOLO finds the object's class → MegaPose coarse search +
  refine inside that box (~1s). Following frames: refine only, from the
  previous pose. A low pose score drops the track and re-detects.
- RGB only (no depth sensor): distance comes from the model's known real
  size, so a wrong `scale` means a wrong distance. Rotationally symmetric
  objects (a mug without its handle visible) have an ambiguous spin angle.
