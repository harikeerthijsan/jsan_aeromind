# Start the AeroMind GPU worker: runs the Railway app's detection, 3D reconstruction and vision-model work on this PC.
# Settings: backend/worker.env (shared database + bucket) on top of backend/.env (YOLO, COLMAP, Ollama).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path worker.env)) { throw 'backend/worker.env is missing: copy worker.env.example and fill it in' }
$env:AEROMIND_ROLE = 'worker'   # writes production storage keys; backend/storage.env (local dev namespace) is not loaded
foreach ($line in Get-Content worker.env) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2].Trim().Trim('"') }
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
while ($true) {   # restart after a crash or a lost connection
    & $python -m app.worker
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "Worker exited ($LASTEXITCODE); restarting in 10 s" -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
