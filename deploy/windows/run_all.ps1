# Kassa Sentinel - one command for L2: preflight -> start -> wait READY -> print URL and acceptance commands.
#   powershell -ExecutionPolicy Bypass -File deploy\windows\run_all.ps1 [-Named] [-NoTunnel]
param([switch]$Named, [switch]$NoTunnel, [int]$WaitSec = 1200)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_env.ps1"
Require-Venv
Set-Location $Repo
$startArgs = @()
if ($Named) { $startArgs += "-Named" }
if ($NoTunnel) { $startArgs += "-NoTunnel" }
& powershell -ExecutionPolicy Bypass -NoProfile -File (Join-Path $PSScriptRoot "start.ps1") @startArgs
if ($LASTEXITCODE -ne 0) { exit 1 }
$deadline = (Get-Date).AddSeconds($WaitSec)
$st = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
    $raw = & $Py deploy\supervisor.py status --json 2>$null
    try { $st = ($raw | Out-String | ConvertFrom-Json) } catch { continue }
    Write-Host ("{0} phase={1} detail={2}" -f (Get-Date -Format T), $st.phase, $st.detail)
    if ($st.phase -in @("crash_loop", "failed", "ready_timeout")) { Write-Host "NOT READY: $($st.last_error)" -ForegroundColor Red; exit 1 }
    if ($st.ok -and ($NoTunnel -or $Named -or $st.public_url)) { break }
}
if (-not $st -or -not $st.ok) { Write-Host "Not ready after ${WaitSec}s - see runs\deploy\ready.json and api.err.log" -ForegroundColor Red; exit 1 }
$url = if ($st.public_url) { $st.public_url } else { "https://<named-hostname>" }
Write-Host ""
Write-Host "READY. Public URL: $url" -ForegroundColor Green
Write-Host "Measure the tunnel:   .venv\Scripts\python deploy\tunnel_probe.py real --url $url --mb 30"
Write-Host "A16 acceptance:       .venv\Scripts\python -m eval.jury_sim --url $url --key <key> --videos data\videos --n 6"
Write-Host "Status / stop:        deploy\windows\status.ps1  /  deploy\windows\stop.ps1"
