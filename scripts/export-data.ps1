<#
Packs this PC's TVASTA work into one zip for another PC: the database (3D
models, assets, procedures, inspections) and the data\ folder (GLB files,
component mappings, pose features, station settings / locks, learned
references, reports). Camera calibration is NOT included: it belongs to this
camera - calibrate again on the other PC.

    powershell -ExecutionPolicy Bypass -File scripts\export-data.ps1
    -> tvasta-export-<date>.zip in the repository folder
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$work = Join-Path $env:TEMP "tvasta-export-$stamp"
New-Item -ItemType Directory -Force $work | Out-Null

Write-Host "==> Database dump" -ForegroundColor Cyan
docker exec tvasta-postgres pg_dump -U tvasta -d tvasta --clean --if-exists -f /tmp/tvasta.sql
if ($LASTEXITCODE -ne 0) { Write-Host "pg_dump failed - is the tvasta-postgres container running?" -ForegroundColor Red; exit 1 }
docker cp tvasta-postgres:/tmp/tvasta.sql (Join-Path $work "tvasta.sql") | Out-Null

Write-Host "==> data folder" -ForegroundColor Cyan
$data = Join-Path $Root "data"
$skip = @("calibration", "debug_frames", "videos")  # camera-specific, or scratch
Get-ChildItem $data -Directory | Where-Object { $skip -notcontains $_.Name } | ForEach-Object {
    Copy-Item $_.FullName -Destination (Join-Path $work "data\$($_.Name)") -Recurse
}

$zip = Join-Path $Root "tvasta-export-$stamp.zip"
Compress-Archive -Path (Join-Path $work "*") -DestinationPath $zip -Force
Remove-Item $work -Recurse -Force
Write-Host "`nWrote $zip - copy it to the other PC and run scripts\import-data.ps1 <zip> there." -ForegroundColor Green
