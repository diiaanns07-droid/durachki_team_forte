# Kassa Sentinel - setup on the Windows laptop (owner: A17). ASCII-only.
# The working venv and the L1 lock (tools\dataprep\requirements-l1.lock.txt) are the source of truth:
# NO blind upgrades. Existing .venv is never re-synced; missing runtime packages are added only with
# -InstallMissing and only under "-c <L1 lock>" (pinned versions cannot change).
#   powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1                 # check only
#   powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1 -InstallMissing # + fastapi/uvicorn/...
#   powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1 -Fresh          # new machine: venv from L1 lock
# GPU probe allocates VRAM: run setup only while the service is stopped.
param(
    [switch]$Fresh,
    [switch]$InstallMissing,
    [switch]$HashWeights,        # full sha256 of every weight file during adopt (slow, once)
    [switch]$SkipFrontend,
    [string]$PythonVersion = "3.11"
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_env.ps1"
Set-Location $Repo
function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "XX  $msg" -ForegroundColor Red; exit 1 }
$Lock = "tools\dataprep\requirements-l1.lock.txt"
if (-not (Test-Path $Lock)) { Fail "L1 lock not found: $Lock (merge branch data/labels)" }

if (-not (Test-Path $Py)) {
    if (-not $Fresh) { Fail "No .venv. Use the L1 venv, or run with -Fresh on a new machine." }
    Step "new venv from the L1 lock (exact versions, torch from the cu128 index)"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv not found: https://docs.astral.sh/uv/" }
    uv python install $PythonVersion
    uv venv .venv --python $PythonVersion
    uv pip install --python $Py -r $Lock --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { Fail "install from L1 lock failed" }
}

Step "envcheck: venv vs L1 lock + module requirements (no installs)"
& $Py deploy\envcheck.py
if ($LASTEXITCODE -ne 0) {
    $missing = (& $Py deploy\envcheck.py --json | Out-String | ConvertFrom-Json).missing_runtime
    if ($InstallMissing -and $missing) {
        Step "install ONLY missing runtime packages under -c L1 lock: $($missing -join ', ')"
        uv pip install --python $Py @missing -c $Lock
        & $Py deploy\envcheck.py
        if ($LASTEXITCODE -ne 0) { Fail "still blocking after install - see envcheck output (do not upgrade pinned packages)" }
    } else { Fail "environment does not match the L1 lock (see BLOCK lines). Missing runtime only? re-run with -InstallMissing" }
}

Step "GPU probe on THIS machine (service must be stopped)"
& $Py deploy\gpu_probe.py --expect-cuda 12.8 --min-vram-gib 7.5
if ($LASTEXITCODE -ne 0) { Fail "GPU probe failed (runs\deploy\gpu_probe.json)" }

Step "weights: adopt the existing offline cache (weights\hf\hub) -> deploy\weights.lock.json, then verify"
$adoptArgs = @("deploy\weights.py", "adopt")
if ($HashWeights) { $adoptArgs += "--hash" }
& $Py @adoptArgs
if ($LASTEXITCODE -ne 0) { Fail "required weights missing/incomplete (see [adopt] FAIL lines)" }
& $Py deploy\weights.py verify --fast
if ($LASTEXITCODE -ne 0) { Fail "weights verify failed" }
& $Py deploy\weights.py licenses | Out-Null

Step "deploy\.env (secrets stay local, never in git)"
if (-not (Test-Path "deploy\.env")) {
    Copy-Item "deploy\.env.example" "deploy\.env"
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = ([Convert]::ToBase64String($bytes)) -replace '[+/=]', ''
    (Get-Content "deploy\.env") -replace '^SENTINEL_API_KEYS=.*$', "SENTINEL_API_KEYS=$key" | Set-Content "deploy\.env" -Encoding ASCII
    Write-Host "Generated API key -> deploy\.env (not printed). Hand it to the jury out of band."
}
if (-not $SkipFrontend -and (Test-Path "frontend\package.json") -and -not (Test-Path "frontend\dist\index.html")) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Step "frontend build (A15)"; Push-Location frontend; if (Test-Path "package-lock.json") { npm ci } else { npm install }; npm run build; Pop-Location
    } else { Write-Host "npm not found - UI not built" -ForegroundColor Yellow }
}
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) { Write-Host "cloudflared not found: winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow }
Step "done. Next: deploy\windows\offline_smoke.ps1 -Firewall (as admin), then deploy\windows\run_all.ps1"
