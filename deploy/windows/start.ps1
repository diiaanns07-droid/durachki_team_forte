# Kassa Sentinel - start the service in the background (owner: A17). ASCII-only.
#   powershell -ExecutionPolicy Bypass -File deploy\windows\start.ps1 [-Named] [-NoTunnel] [-Port 8000]
# preflight (blocking) -> deploy\supervisor.py run (hidden, ONE instance): one uvicorn worker, restart on
# crash/hang (never a second server, never kills foreign processes), READY only after /readyz=200 and a
# real trial job, then cloudflared. Status: status.ps1, stop: stop.ps1. No global firewall/power changes.
param(
    [int]$Port = 0,
    [string]$App = "",
    [switch]$Named,
    [switch]$NoTunnel,
    [switch]$SkipPreflight,      # emergency only - not for the jury run
    [string]$BindHost = "",      # kept for compatibility; host comes from deploy\config\deploy.yaml
    [int]$ReadyTimeoutSec = 0    # kept for compatibility; see service.ready_timeout_s in deploy.yaml
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_env.ps1"
Require-Venv
Set-Location $Repo
if ($Port -gt 0) { $env:SENTINEL_PORT = "$Port" }
if (-not $SkipPreflight) {
    & $Py deploy\preflight.py
    if ($LASTEXITCODE -ne 0) { Write-Host "PREFLIGHT FAILED - service NOT started (runs\deploy\preflight.json)" -ForegroundColor Red; exit 1 }
}
# Start-Process -ArgumentList does not quote elements with spaces (PS 5.1 and 7): quote explicitly.
function Quote-Arg([string]$a) { if ($a -match '[\s"]') { '"' + ($a -replace '"', '\"') + '"' } else { $a } }
$supArgs = @((Join-Path $Repo "deploy\supervisor.py"), "run")
if ($Named) { $supArgs += "--named" }
if ($NoTunnel) { $supArgs += "--no-tunnel" }
if ($Port -gt 0) { $supArgs += @("--port", "$Port") }
if ($App) { $supArgs += @("--app", $App) }
New-Item -ItemType Directory -Force -Path $env:SENTINEL_LOG_DIR | Out-Null
$p = Start-Process -FilePath $Py -ArgumentList ($supArgs | ForEach-Object { Quote-Arg $_ }) -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $env:SENTINEL_LOG_DIR "supervisor.console.log") `
    -RedirectStandardError (Join-Path $env:SENTINEL_LOG_DIR "supervisor.console.err.log")
Start-Sleep -Seconds 3
if ($p.HasExited -and $p.ExitCode -eq 3) { Write-Host "Already running - see deploy\windows\status.ps1" -ForegroundColor Yellow; exit 0 }
if ($p.HasExited) { Write-Host "Supervisor exited with code $($p.ExitCode) - see runs\deploy\supervisor.log" -ForegroundColor Red; exit 1 }
Write-Host "Supervisor started (pid $($p.Id)). Warmup + trial job take a few minutes. Check: deploy\windows\status.ps1"
