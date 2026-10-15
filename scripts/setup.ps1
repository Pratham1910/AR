<#
TVASTA one-time setup on a Windows PC (backend, frontend, database).
The GPU pose service is set up separately inside WSL: pose_service/setup_wsl.sh.

Run from the repository root, in PowerShell:
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

Safe to run again: every step checks what is already there.
Prerequisites it checks for (install them first, see docs/setup.md):
    Python 3.11, Node.js 18+, Docker Desktop (running), WSL 2.
#>
param(
    [switch]$SkipDocker,   # PostgreSQL + MinIO already provided some other way
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }
function Ok($message) { Write-Host "    ok: $message" -ForegroundColor Green }
function Fail($message) { Write-Host "`nSETUP STOPPED: $message" -ForegroundColor Red; exit 1 }

# -- 1. Prerequisites -----------------------------------------------------
Step "Checking prerequisites"

# Python 3.11 exactly (numpy 1.26 / open3d 0.18 / mediapipe wheels): the py launcher first, then python.
$Python = $null
foreach ($candidate in @(@("py", "-3.11"), @("python"))) {
    try {
        $version = & $candidate[0] $candidate[1..9] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($version -eq "3.11") { $Python = $candidate; break }
    } catch {}
}
if ($null -eq $Python) { Fail "Python 3.11 not found. Install it from python.org (tick 'Add python.exe to PATH') and run again." }
Ok "Python 3.11 ($($Python -join ' '))"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail "Node.js not found. Install Node.js 20 LTS from nodejs.org and run again." }
$nodeMajor = [int]((node -v).TrimStart("v").Split(".")[0])
if ($nodeMajor -lt 18) { Fail "Node.js $(node -v) is too old; install Node.js 20 LTS." }
Ok "Node.js $(node -v)"

if (-not $SkipDocker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "Docker not found. Install Docker Desktop, start it, and run again (or use -SkipDocker with your own PostgreSQL)." }
    docker info *> $null
    if ($LASTEXITCODE -ne 0) { Fail "Docker is installed but not running. Start Docker Desktop and run again." }
    Ok "Docker running"
}

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host "    warning: WSL not found - the GPU pose service (model-based tracking) needs it. See docs/setup.md." -ForegroundColor Yellow
} else { Ok "WSL present" }

# -- 2. Database + object storage -----------------------------------------
if (-not $SkipDocker) {
    Step "Starting PostgreSQL and MinIO (docker compose)"
    Push-Location $Root
    docker compose up -d postgres minio
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "docker compose failed (see above)." }
    Pop-Location
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        docker exec tvasta-postgres pg_isready -U tvasta *> $null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { Fail "PostgreSQL did not become ready (docker logs tvasta-postgres)." }
    Ok "PostgreSQL ready on 5432, MinIO on 9000"
}

# -- 3. Backend Python environment ----------------------------------------
Step "Backend: Python environment (backend\.venv)"
$VenvPython = Join-Path $Backend ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $Python[0] $Python[1..9] -m venv (Join-Path $Backend ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $Backend "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "pip install -r backend\requirements.txt failed (see above)." }

# mediapipe (hand masks) without its declared deps: they would add a second OpenCV build.
& $VenvPython -m pip install --no-deps "mediapipe==1.1.0" "absl-py~=2.3" "flatbuffers~=25.9" "sounddevice~=0.5" --quiet
if ($LASTEXITCODE -ne 0) { Fail "mediapipe install failed." }

# ultralytics pulls opencv-python next to opencv-contrib-python-headless; both
# write the same cv2 package and the result depends on install order. Keep
# exactly one: the contrib build (it has ArUco, which marker tracking needs).
& $VenvPython -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python *> $null
& $VenvPython -m pip install --force-reinstall --no-deps "opencv-contrib-python-headless==4.10.0.84" --quiet
& $VenvPython -c "import cv2, sys; sys.exit(0 if hasattr(cv2, 'aruco') and hasattr(cv2.aruco, 'ArucoDetector') else 1)"
if ($LASTEXITCODE -ne 0) { Fail "OpenCV check failed (cv2.aruco missing)." }
Ok "Python packages installed, OpenCV contrib in place"

# Model files the backend needs on disk.
$HandModel = Join-Path $Root "models\mediapipe\hand_landmarker.task"
if (-not (Test-Path $HandModel)) {
    New-Item -ItemType Directory -Force (Split-Path $HandModel) | Out-Null
    Invoke-WebRequest -UseBasicParsing -OutFile $HandModel `
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
}
Ok "hand landmark model present"
$Yolo = Join-Path $Backend "yolov8n-seg.pt"
if (-not (Test-Path $Yolo)) {
    Invoke-WebRequest -UseBasicParsing -OutFile $Yolo "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt"
}
Ok "YOLO segmentation weights present"

# -- 4. Database schema ---------------------------------------------------
Step "Database schema (alembic upgrade head)"
Push-Location $Backend
& $VenvPython -m alembic upgrade head
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Fail "Database migration failed - is PostgreSQL running on localhost:5432 (user/password tvasta)?" }
Ok "schema up to date"

# -- 5. Frontend ----------------------------------------------------------
if (-not $SkipFrontend) {
    Step "Frontend packages (npm ci)"
    Push-Location $Frontend
    npm ci --no-audit --no-fund
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "npm ci failed (see above)." }
    Ok "frontend ready"
}

# -- 6. Self-test ---------------------------------------------------------
Step "Quick self-test (backend imports)"
Push-Location $Backend
& $VenvPython -c "import app.main; print('backend imports OK')"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Fail "The backend does not import - see the error above." }

Write-Host "`nWindows side done." -ForegroundColor Green
Write-Host "Next:"
Write-Host "  1. GPU pose service, once, inside WSL (Ubuntu 22.04):"
Write-Host "       bash pose_service/setup_wsl.sh      (run it from the repo folder inside WSL)"
Write-Host "  2. Every day:  powershell -ExecutionPolicy Bypass -File scripts\start.ps1"
Write-Host "  3. Check:      powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1"
