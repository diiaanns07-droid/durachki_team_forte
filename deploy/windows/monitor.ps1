# Kassa Sentinel - resource log every N seconds (owner: A17). ASCII-only. Locale-independent (CIM, nvidia-smi).
#   powershell -ExecutionPolicy Bypass -File deploy\windows\monitor.ps1 [-IntervalSec 30]
param([int]$IntervalSec = 30)
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Csv = Join-Path $Repo "runs\deploy\monitor.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $Csv) | Out-Null
if (-not (Test-Path $Csv)) { "time,cpu_pct,ram_used_gb,ram_total_gb,gpu_util_pct,vram_used_mib,vram_total_mib,gpu_temp_c,gpu_power_w,python_ws_mb" | Set-Content $Csv }
while ($true) {
    $cpu = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 0)
    $os = Get-CimInstance Win32_OperatingSystem
    $tot = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1); $used = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
    $g = "", "", "", "", ""
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | Select-Object -First 1).Split(",") | ForEach-Object { $_.Trim() }
    }
    $ws = [math]::Round(((Get-Process python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum) / 1MB, 0)
    $line = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f (Get-Date -Format s), $cpu, $used, $tot, $g[0], $g[1], $g[2], $g[3], $g[4], $ws
    Add-Content $Csv $line; Write-Host $line
    Start-Sleep -Seconds $IntervalSec
}
