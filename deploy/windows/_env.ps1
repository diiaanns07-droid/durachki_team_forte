# Kassa Sentinel - shared env loader (dot-source: . "$PSScriptRoot\_env.ps1"). ASCII-only.
# Loads deploy\.env into THIS process (children inherit). Secrets stay in env, never printed, never in git.
# /app/... container paths are mapped into the repo. Runtime is always offline (HF_HUB_OFFLINE=1).
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$EnvFile = Join-Path $Repo "deploy\.env"
if (Test-Path $EnvFile) {
    foreach ($raw in Get-Content $EnvFile) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
        $k, $v = $line.Split("=", 2)
        $v = ($v -replace '\s+#.*$', '').Trim().Trim('"').Trim("'")
        if ($v.StartsWith("/app/")) { $v = Join-Path $Repo ($v.Substring(5) -replace "/", "\") }
        if ($v -ne "") { [Environment]::SetEnvironmentVariable($k.Trim(), $v, "Process") }
    }
}
# Optional profile (no secrets): SENTINEL_PROFILE=pose_rules -> deploy\profiles\pose_rules.env on top of deploy\.env
if ($env:SENTINEL_PROFILE) {
    $ProfileFile = Join-Path $Repo ("deploy\profiles\" + $env:SENTINEL_PROFILE + ".env")
    if (-not (Test-Path $ProfileFile)) { Write-Host "Unknown SENTINEL_PROFILE: $env:SENTINEL_PROFILE" -ForegroundColor Red; exit 1 }
    foreach ($raw in Get-Content $ProfileFile) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
        $k, $v = $line.Split("=", 2)
        $v = ($v -replace '\s+#.*$', '').Trim().Trim('"').Trim("'")
        if ($v.StartsWith("/app/")) { $v = Join-Path $Repo ($v.Substring(5) -replace "/", "\") }
        if ($v -ne "") { [Environment]::SetEnvironmentVariable($k.Trim(), $v, "Process") }
    }
}
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"; $env:DO_NOT_TRACK = "1"; $env:SENTINEL_WORKERS = "1"; $env:PYTHONUTF8 = "1"
if (-not $env:SENTINEL_WEIGHTS) { $env:SENTINEL_WEIGHTS = Join-Path $Repo "weights" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $env:SENTINEL_WEIGHTS "hf" }
if (-not $env:HF_HUB_CACHE) { $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub" }
if (-not $env:SENTINEL_LOG_DIR) { $env:SENTINEL_LOG_DIR = Join-Path $Repo "runs\deploy" }
function Require-Venv { if (-not (Test-Path $Py)) { Write-Host "No .venv - run deploy\windows\setup.ps1" -ForegroundColor Red; exit 1 } }
