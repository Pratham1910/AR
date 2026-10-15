# Model-based (CAD) pose service

MegaPose (via [HappyPose](https://github.com/agimus-project/happypose)) matches
a registered 3D model's own mesh against a webcam frame and returns its full
6DoF pose. It runs in WSL2 with CUDA; the Windows backend calls it at
`POSE_SERVICE_URL` (default `http://127.0.0.1:8765`) from the AR session
(`POST /api/vision/ar-session/frame`, mode "model").

## One-time setup (WSL, Ubuntu 22.04)

```bash
bash pose_service/setup_wsl.sh      # from the repository folder inside WSL
```

Installs HappyPose at a pinned commit with the pinned package versions
(`requirements-lock.txt`, frozen from the working PC), the MegaPose models,
and the FastSAM + DINOv2 weights for finding objects by their 3D model; then
self-tests CUDA. See `docs/setup.md` for the whole PC setup.

Larger, somewhat more accurate variants: `FastSAM-x.pt` and
`dinov2_vitl14_pretrain.pth` (1.2 GB), selected with
`TVASTA_FASTSAM_WEIGHTS=FastSAM-x.pt TVASTA_DINO_MODEL=dinov2_vitl14`.

## Run

```bash
bash pose_service/start.sh      # from the repository folder inside WSL (or scripts\start.ps1 on Windows)
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
