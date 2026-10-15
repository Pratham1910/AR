# Setting TVASTA up on a new PC

Windows (steps 1-6) or Linux (step 6b). Four commands on a prepared PC; everything else is checked and explained by
the scripts. About 1 hour the first time (mostly downloads).

```
Windows                                         WSL 2 (Ubuntu 22.04)
┌─────────────────────────────────────────┐     ┌──────────────────────────────┐
│ Browser  ── http://127.0.0.1:5173       │     │ Pose service :8765 (GPU)     │
│   frontend (Vite, React, Three.js)      │     │  MegaPose · FastSAM · DINOv2 │
│ Backend  ── http://127.0.0.1:8000       │────▶│  ~/tvasta-pose/              │
│   FastAPI, OpenCV, YOLO, MediaPipe      │     └──────────────────────────────┘
│ Docker: PostgreSQL :5432, MinIO :9000   │
└─────────────────────────────────────────┘
```

## 1. What the PC needs

| | Requirement | Notes |
|---|---|---|
| GPU | **NVIDIA**, 8 GB+ VRAM (reference: RTX 4090) | Model-based tracking (MegaPose) needs CUDA. Without it the app runs, but only marker / markerless modes. |
| OS | Windows 10 22H2 or Windows 11, 64-bit | WSL 2 required. |
| RAM / disk | 16 GB (32 recommended) / 30 GB free | The pose environment + weights are ~12 GB. |
| Camera | Logitech BRIO (reference) or any UVC webcam | Calibrate each camera once (step 6). |
| Browser | Chrome or Edge | Camera permission for 127.0.0.1. |

## 2. Install these first (once)

| Software | Version | Get it |
|---|---|---|
| NVIDIA driver (Windows) | recent Game Ready / Studio (reference 617.14) | nvidia.com/drivers — **not** inside WSL |
| WSL 2 + Ubuntu 22.04 | — | PowerShell as admin: `wsl --install -d Ubuntu-22.04`, reboot, create the Linux user |
| Docker Desktop | current, "Use WSL 2 based engine" on | docker.com |
| Python | **3.11.x** (exactly 3.11) | python.org — tick *Add python.exe to PATH* |
| Node.js | 20 LTS (18+ works) | nodejs.org |
| Git | any | git-scm.com |

## 3. Get the code

```powershell
git clone https://github.com/Pratham1910/AR.git TVASTA
cd TVASTA
```

(If the latest work isn't pushed yet, copy the whole project folder instead —
without `backend\.venv` and `frontend\node_modules`, which are rebuilt.)

## 4. Windows side — one command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

It checks the prerequisites and stops with the fix if one is missing, then:
starts PostgreSQL + MinIO (docker compose), creates `backend\.venv` with all
Python packages (and the OpenCV / MediaPipe fix-ups below), downloads the hand
and YOLO models, creates the database schema, installs the frontend packages,
and self-tests the backend. Safe to run again.

## 5. GPU pose service — one command inside WSL

Open Ubuntu (Start menu), go to the project folder through `/mnt/c/...`, and run:

```bash
cd /mnt/c/Users/<you>/Desktop/TVASTA      # wherever you cloned it
bash pose_service/setup_wsl.sh
```

It checks the GPU is visible in WSL, installs system packages, `uv`, HappyPose
at the **pinned commit**, the **pinned package versions**
(`pose_service/requirements-lock.txt`, frozen from the working PC), the
MegaPose / FastSAM / DINOv2 weights, and self-tests CUDA. 15–40 minutes.

**Zero-effort alternative — clone the working WSL instead** (identical
environment, no downloads; ~15–20 GB file):

```powershell
# on the working PC
wsl --shutdown
wsl --export Ubuntu-22.04 D:\tvasta-wsl.tar
# on the new PC (after installing WSL 2)
wsl --import Ubuntu-22.04 C:\WSL\Ubuntu-22.04 D:\tvasta-wsl.tar
```

## 6. Run, check, first use

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1    # everything, in separate windows; opens the app
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1   # checks every part, prints the fix for each problem
```

First use on the new PC:
1. **Calibrate the camera** — AR view → *Calibrate camera*: print a checkerboard
   (9×6 inner corners), capture 15–20 frames from different angles, *Compute*.
   Without it the intrinsics are assumed (82° field of view) and accuracy drops.
2. Set each model's **resting side** and, for symmetric parts, its **pose
   features** (`data/component_registry/<model id>.json`, see
   `docs/phase8/report.md` §5).
3. For a fixed bench: line the part up once and *Save guide view*.

## 6b. Linux instead of Windows (Ubuntu, Debian, Arch / Omarchy, Fedora)

Simpler than Windows: no WSL — the GPU pose service runs natively next to
the backend.

| | Ubuntu / Debian | Arch / Omarchy |
|---|---|---|
| NVIDIA driver | `sudo ubuntu-drivers install`, reboot | `nvidia` or `nvidia-open` (+ `nvidia-utils`), reboot — Omarchy offers it at install |
| Docker | docs.docker.com/engine/install/ubuntu | `sudo pacman -S docker docker-compose && sudo systemctl enable --now docker` |
| Docker without sudo | `sudo usermod -aG docker $USER`, log out/in | same |
| Node.js 20 | NodeSource packages (Ubuntu's own is too old on 22.04) | `sudo pacman -S nodejs npm` |
| Python 3.11 | **nothing to do** — `uv` installs it (system 3.10 / 3.12 is fine) | **nothing to do** — Arch's 3.13 is left alone |
| Browser | Chrome / Chromium / Firefox | same (Hyprland/Wayland: camera permission works through the browser) |

No CUDA toolkit is needed on either: PyTorch brings its own CUDA 12.4 runtime.

```bash
git clone https://github.com/Pratham1910/AR.git TVASTA && cd TVASTA
bash scripts/setup_linux.sh            # backend (Python 3.11 via uv), frontend, database
bash pose_service/setup_wsl.sh         # GPU pose service — same script, works natively
bash scripts/start_linux.sh            # everything in the background, logs in .run/*.log
bash scripts/doctor.sh                 # check; prints the fix for each problem
bash scripts/stop_linux.sh             # stop
```

Options: `--skip-docker` (own PostgreSQL), `--skip-frontend`,
`--no-system-packages` (no sudo; install git, curl, unzip, a C compiler,
libGL/libEGL and glib yourself). Moving data: `export-data.ps1` /
`import-data.ps1` are PowerShell — on Linux the same steps by hand:
`docker exec tvasta-postgres pg_dump -U tvasta -d tvasta --clean --if-exists > tvasta.sql`
and copy `data/` (without `data/calibration`); restore with
`docker exec -i tvasta-postgres psql -U tvasta -d tvasta < tvasta.sql`.

Arch / Omarchy notes: Python 3.13 and new system libraries are why the
backend never uses the system Python (pinned wheels like numpy 1.26 and
open3d 0.18 exist only up to 3.11/3.12). If the pose service's Panda3D
renderer fails to start under Wayland, it is looking for an EGL display:
`sudo pacman -S mesa libglvnd` (installed by the setup script) and, with the
NVIDIA driver, `nvidia-utils` provides the EGL vendor library.

## 7. Moving your existing work (models, procedures, mappings, reports)

```powershell
# working PC
powershell -ExecutionPolicy Bypass -File scripts\export-data.ps1       # -> tvasta-export-<date>.zip
# new PC (after step 4)
powershell -ExecutionPolicy Bypass -File scripts\import-data.ps1 tvasta-export-<date>.zip
```

The export holds the database and `data\` (GLB files, component mappings, pose
features, station settings / locks / guide views, learned references, reports).
Camera calibration is deliberately left out — it belongs to one camera.

## 8. Configuration

Nothing is required: the defaults match the scripts. To change something,
create `backend\.env` (read from the backend's folder) with any of:

| Setting | Default | What |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://tvasta:tvasta@localhost:5432/tvasta` | PostgreSQL (docker-compose user/password `tvasta`) |
| `POSE_SERVICE_URL` | `http://127.0.0.1:8765` | GPU pose service |
| `CAMERA_ASSUMED_HFOV_DEG` | `82` | Used only until the camera is calibrated |
| `CAMERA_CALIBRATION_PATH` | (unset → `data/calibration/camera.json`) | Calibration file |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `localhost:9000` / `tvasta` / `tvasta12345` | Evidence storage |
| `HAND_MASK_METHOD` | `both` | `mediapipe`, `yolo`, `both`, `off` |
| `JWT_SECRET_KEY` | dev default | **Change for any shared deployment** |
| `AR_*`, `AR_ORIENTATION_*`, `AR_GUIDE_*` | see `backend/app/core/config.py` | Tracking tuning |

Frontend: `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`) in `frontend\.env`
if the backend runs elsewhere. Pose service: `TVASTA_POSE_HOME`,
`TVASTA_FASTSAM_WEIGHTS`, `TVASTA_DINO_MODEL` (see `pose_service/README.md`).

Ports: 5173 frontend · 8000 backend · 8765 pose service · 5432 PostgreSQL · 9000/9001 MinIO.

## 9. Known problems and their fixes (all seen on the reference PC)

| Symptom | Cause | Fix |
|---|---|---|
| Backend errors "connection refused 5432" | Docker Desktop not running | Start Docker Desktop (`doctor.ps1` says so) |
| Port 8000 answers but isn't TVASTA | another program / container on 8000 | `doctor.ps1` names it; stop it. Always use `127.0.0.1`, not `localhost` |
| "Pose service not reachable" | not started, or the WSL window was closed | `start.ps1` (keeps it in its own window) |
| Pose service: `GPU error` / `cuda: false` | NVIDIA driver updated while WSL ran | `wsl --shutdown`, start again; update the Windows driver if it persists |
| No camera in the dropdown / "camera in use" | another app holds the camera, or Chrome's capture stuck | close Teams/Zoom/Camera app; `chrome://restart`. Don't kill Chrome's capture process by hand |
| `cv2` has no `aruco` / odd OpenCV errors | two OpenCV packages in one venv | re-run `setup.ps1` (it removes the duplicate) |
| MediaPipe pulls a second OpenCV | its declared deps | `setup.ps1` installs it with `--no-deps` |
| `bash: $'\r': command not found` | `.sh` checked out with Windows line endings | `.gitattributes` prevents it; or `sed -i 's/\r$//' pose_service/*.sh` |
| First lock of a new model takes ~10–30 s | templates rendered + MegaPose reloads once per new model | normal; cached afterwards |

## 10. Verify the install

```powershell
cd backend
.venv\Scripts\python -m pytest ..\tests\backend -q      # ~400 tests, ~2 min
cd ..\frontend
npm test; npm run build
```
