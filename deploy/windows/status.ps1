# Kassa Sentinel - service status (owner: A17). ASCII-only. Exit 0 = ready.
. "$PSScriptRoot\_env.ps1"
Require-Venv
Set-Location $Repo
& $Py deploy\supervisor.py status
$code = $LASTEXITCODE
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
}
$log = Join-Path $env:SENTINEL_LOG_DIR "supervisor.log"
if (Test-Path $log) { Write-Host "--- last supervisor log lines"; Get-Content $log -Tail 8 }
exit $code
