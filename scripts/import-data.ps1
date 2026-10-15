<#
Restores a zip made by scripts\export-data.ps1 on this PC (after scripts\setup.ps1).
REPLACES this PC's TVASTA database and the exported data\ subfolders.

    powershell -ExecutionPolicy Bypass -File scripts\import-data.ps1 tvasta-export-20261015-1200.zip
#>
param([Parameter(Mandatory = $true)][string]$Zip)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Zip)) { Write-Host "No such file: $Zip" -ForegroundColor Red; exit 1 }

$answer = Read-Host "This replaces the TVASTA database and data folders on this PC. Type YES to continue"
if ($answer -ne "YES") { Write-Host "Cancelled."; exit 0 }

$work = Join-Path $env:TEMP ("tvasta-import-" + (Get-Date -Format "yyyyMMddHHmmss"))
Expand-Archive -Path $Zip -DestinationPath $work

Write-Host "==> Database" -ForegroundColor Cyan
docker cp (Join-Path $work "tvasta.sql") tvasta-postgres:/tmp/tvasta.sql | Out-Null
docker exec tvasta-postgres psql -q -U tvasta -d tvasta -v ON_ERROR_STOP=1 -f /tmp/tvasta.sql
if ($LASTEXITCODE -ne 0) { Write-Host "Restore failed - see above." -ForegroundColor Red; exit 1 }

Write-Host "==> data folder" -ForegroundColor Cyan
Get-ChildItem (Join-Path $work "data") -Directory | ForEach-Object {
    $target = Join-Path $Root "data\$($_.Name)"
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Copy-Item $_.FullName -Destination $target -Recurse
}
Remove-Item $work -Recurse -Force

Write-Host "==> Database schema (in case this code is newer than the export)" -ForegroundColor Cyan
Push-Location (Join-Path $Root "backend")
& ".venv\Scripts\python.exe" -m alembic upgrade head
Pop-Location

Write-Host "`nImported. Restart the backend, then calibrate this PC's camera (AR view -> Calibrate camera)." -ForegroundColor Green
