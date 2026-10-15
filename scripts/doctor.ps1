<#
Checks every part TVASTA needs and says how to fix what's wrong.
    powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
#>
$Root = Split-Path -Parent $PSScriptRoot
$problems = 0

function Good($what) { Write-Host ("  [ok]   " + $what) -ForegroundColor Green }
function Bad($what, $fix) {
    Write-Host ("  [FAIL] " + $what) -ForegroundColor Red
    Write-Host ("         fix: " + $fix) -ForegroundColor Yellow
    $script:problems++
}
function Warn($what, $fix) {
    Write-Host ("  [warn] " + $what) -ForegroundColor Yellow
    if ($fix) { Write-Host ("         " + $fix) }
}
function Get-Json($url) {
    try { return Invoke-RestMethod -TimeoutSec 4 $url } catch { return $null }
}
function Owner($port) {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $c) { return $null }
    $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    if ($null -eq $p) { return "PID $($c.OwningProcess)" }
    return "$($p.ProcessName) (PID $($p.Id))"
}

Write-Host "TVASTA doctor`n"

# Tools
if (Test-Path (Join-Path $Root "backend\.venv\Scripts\python.exe")) { Good "backend Python environment" } else { Bad "backend\.venv missing" "run scripts\setup.ps1" }
if (Test-Path (Join-Path $Root "frontend\node_modules")) { Good "frontend packages" } else { Bad "frontend\node_modules missing" "run scripts\setup.ps1" }
if (Test-Path (Join-Path $Root "models\mediapipe\hand_landmarker.task")) { Good "hand landmark model" } else { Warn "models\mediapipe\hand_landmarker.task missing (hands found by YOLO only)" "run scripts\setup.ps1" }

# Docker services
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Bad "Docker is not running" "start Docker Desktop"
} else {
    docker exec tvasta-postgres pg_isready -U tvasta *> $null
    if ($LASTEXITCODE -eq 0) { Good "PostgreSQL (tvasta-postgres) accepting connections" } else { Bad "PostgreSQL not ready" "docker compose up -d postgres   (in the repo folder)" }
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:9000/minio/health/live" | Out-Null; Good "MinIO" } catch { Bad "MinIO not reachable on 9000" "docker compose up -d minio" }
}

# Backend
$status = Get-Json "http://127.0.0.1:8000/api/system/status"
if ($null -ne $status) {
    Good "backend on 127.0.0.1:8000"
    if ($status.camera.approximate) { Warn "camera not calibrated (intrinsics assumed: $($status.camera.assumed_hfov_deg) deg)" "AR view -> Calibrate camera (checkerboard), once per camera" } else { Good "camera calibration: $($status.camera.calibration_source)" }
} else {
    $owner = Owner 8000
    if ($owner) { Bad "port 8000 answers but not as the TVASTA backend - taken by $owner" "stop that program (e.g. a docker container), or run the backend on another port" }
    else { Bad "backend not running" "scripts\start.ps1  (or: cd backend; .venv\Scripts\python -m uvicorn app.main:app --port 8000)" }
}

# Pose service
$pose = Get-Json "http://127.0.0.1:8765/health"
if ($null -ne $pose) {
    if ($pose.ok -and $pose.cuda) { Good "pose service on 8765, GPU: $($pose.device), $($pose.objects.Count) model(s) registered" }
    else { Bad "pose service up but GPU unusable: $($pose.gpu_error)" "update the NVIDIA driver on Windows, then 'wsl --shutdown' and start the pose service again" }
} else {
    Bad "pose service not reachable on 8765 (model-based tracking won't work)" "scripts\start.ps1, or in WSL: bash pose_service/start.sh  (first time: bash pose_service/setup_wsl.sh)"
}

# Frontend
try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:5173" | Out-Null; Good "frontend on 127.0.0.1:5173" } catch { Bad "frontend not running" "scripts\start.ps1  (or: cd frontend; npm run dev)" }

# WSL GPU
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    $gpu = wsl nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) { Good "GPU visible in WSL: $gpu" } else { Bad "no GPU inside WSL" "install/update the NVIDIA driver on Windows, run 'wsl --shutdown', reopen WSL" }
} else {
    Bad "WSL not installed" "PowerShell as admin: wsl --install -d Ubuntu-22.04"
}

Write-Host ""
if ($problems -eq 0) { Write-Host "Everything looks fine." -ForegroundColor Green } else { Write-Host "$problems problem(s) - see the fixes above." -ForegroundColor Red }
