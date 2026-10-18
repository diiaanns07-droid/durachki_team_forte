# Kassa Sentinel - locality proof on Windows (owner: A17). ASCII-only.
# REAL worker (backend.pipeline.run -> A03 spawn worker) on prepared local weights, HF offline,
# netguard=block in the process and its workers. With -Firewall (admin): a TEMPORARY outbound block rule
# for python.exe only, plus an OS-level self-test (python.exe without netguard must fail to reach the
# internet). The rule is removed in finally (unless -KeepFirewall). The public HTTP path (cloudflared) is
# checked separately: deploy\tunnel_probe.py. Pitch log: runs\deploy\offline_proof_<time>.md
#   powershell -ExecutionPolicy Bypass -File deploy\windows\offline_smoke.ps1 [-Firewall] [-- extra args]
param([switch]$Firewall, [switch]$KeepFirewall, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_env.ps1"
Require-Venv
Set-Location $Repo
$env:SENTINEL_MOCK = "0"
$smokeArgs = @("deploy\offline_worker_smoke.py")
if ($Rest) { $smokeArgs += $Rest }
$fw = Join-Path $PSScriptRoot "firewall_python.ps1"
if ($Firewall) {
    & powershell -ExecutionPolicy Bypass -NoProfile -File $fw -Action Enable
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not add the firewall rule (run as Administrator)" -ForegroundColor Red; exit 1 }
    $env:SENTINEL_FIREWALL_RULE = "group KassaSentinel-NoEgress: outbound Block for venv python.exe + base python.exe, loopback open, temporary"
    $smokeArgs += @("--egress-selftest", "--require-egress-blocked")
}
$code = 1
try {
    & $Py @smokeArgs
    $code = $LASTEXITCODE
} finally {
    if ($Firewall -and -not $KeepFirewall) { & powershell -ExecutionPolicy Bypass -NoProfile -File $fw -Action Disable | Out-Null }
}
exit $code
