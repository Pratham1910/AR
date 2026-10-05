# Model-based (CAD) pose service

MegaPose (via [HappyPose](https://github.com/agimus-project/happypose)) matches
a registered 3D model's own mesh against a webcam frame and returns its full
6DoF pose. It runs in WSL2 with CUDA; the Windows backend calls it at
`POSE_SERVICE_URL` (default `http://127.0.0.1:8765`) from the AR session
(`POST /api/vision/ar-session/frame`, mode "model").

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

Finding objects by their 3D model (no object class) also needs FastSAM and
DINOv2. `--no-deps`: ultralytics would otherwise add a second OpenCV build next
to HappyPose's.

```bash
uv pip install --python .venv/bin/python --no-deps ultralytics==8.3.10 ultralytics-thop==2.2.2 py-cpuinfo==9.0.0
mkdir -p ~/tvasta-pose/weights && cd ~/tvasta-pose/weights
curl -LO https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt
curl -O https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
curl -L -o dinov2.zip https://github.com/facebookresearch/dinov2/zipball/main && unzip -q dinov2.zip && mv facebookresearch-dinov2-* dinov2
```

Larger, somewhat more accurate variants: `FastSAM-x.pt` and
`dinov2_vitl14_pretrain.pth` (1.2 GB), selected with
`TVASTA_FASTSAM_WEIGHTS=FastSAM-x.pt TVASTA_DINO_MODEL=dinov2_vitl14`.

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
- Finding the object (first frame, and after tracking is lost), `POST /detect`:
  no object class — the mesh is rendered from 42 viewpoints once
  (`render_templates.py`, ~10 s, cached in `meshes/templates/`), FastSAM splits
  the camera image into candidate regions, and the region whose DINOv2
  descriptor best matches the renders is the object (the CNOS method). The
  older "Object class" option (YOLO) is still available in the app.
- Then MegaPose coarse search + refine inside that box (~1s). Given the
  detected outline, the 5 best candidate poses are refined and the one whose
  silhouette overlaps the outline best is kept: MegaPose's own score can't
  tell a bottle from the same bottle upside down (same box), the outline can. Following
  frames: refine only, from the previous pose; the detector runs again only
  if tracking is lost.
- RGB only (no depth sensor): distance comes from the model's known real
  size, so a wrong `scale` means a wrong distance. Rotationally symmetric
  objects (a mug without its handle visible) have an ambiguous spin angle.
