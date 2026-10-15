<#
Starts everything TVASTA needs, each in its own window, then opens the app:
    PostgreSQL + MinIO (docker), the GPU pose service (WSL), the backend, the frontend.

    powershell -ExecutionPolicy Bypass -File scripts\start.ps1
    (add -NoPose to skip the pose service, e.g. on a PC without an NVIDIA GPU)
#>
param([switch]$NoPose, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$VenvPython = Join-Path $Backend ".venv\Scripts\python.exe"

function Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function PortBusy($port) { return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) }

if (-not (Test-Path $VenvPython)) { Write-Host "Run scripts\setup.ps1 first." -ForegroundColor Red; exit 1 }

Step "Database + storage"
docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "Docker Desktop is not running - start it and run this again." -ForegroundColor Red; exit 1 }
Push-Location $Root
docker compose up -d postgres minio | Out-Null
Pop-Location

if (-not $NoPose) {
    Step "GPU pose service (WSL)"
    if (PortBusy 8765) {
        Write-Host "    already running on 8765"
    } else {
        $wslRepo = (wsl wslpath -a "$Root").Trim()
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='TVASTA pose service'; wsl bash '$wslRepo/pose_service/start.sh'"
    }
}

Step "Backend (http://127.0.0.1:8000)"
if (PortBusy 8000) {
    Write-Host "    port 8000 is already taken - if it isn't the TVASTA backend, stop that program (scripts\doctor.ps1 shows which)." -ForegroundColor Yellow
} else {
    Start-Process powershell -WorkingDirectory $Backend -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle='TVASTA backend'; & '$VenvPython' -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
}

Step "Frontend (http://127.0.0.1:5173)"
if (PortBusy 5173) {
    Write-Host "    already running on 5173"
} else {
    Start-Process powershell -WorkingDirectory $Frontend -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle='TVASTA frontend'; npm run dev -- --host 127.0.0.1"
}

if (-not $NoBrowser) {
    Step "Waiting for the app"
    for ($i = 0; $i -lt 60; $i++) {
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:5173" | Out-Null; break } catch { Start-Sleep -Seconds 1 }
    }
    Start-Process "http://127.0.0.1:5173"
}
Write-Host "`nStarted. The pose service takes ~1 min to load MegaPose. Check everything with scripts\doctor.ps1" -ForegroundColor Green
