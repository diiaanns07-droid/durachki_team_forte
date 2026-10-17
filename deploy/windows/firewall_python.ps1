# Kassa Sentinel - TEMPORARY per-program outbound block for python.exe (owner: A17). ASCII-only. Needs admin.
# NOT a global firewall change: only outbound traffic of these python.exe files is blocked:
#   - .venv\Scripts\python.exe (venv launcher) and
#   - sys._base_executable (the real interpreter that the launcher and multiprocessing spawn run).
# Note: the base interpreter may be shared by other venvs on this PC - they are blocked too while enabled.
# Loopback stays open (RemoteAddress excludes 127.0.0.0/8 and ::1); cloudflared.exe is another program.
#   powershell -ExecutionPolicy Bypass -File deploy\windows\firewall_python.ps1 -Action Enable|Disable|Status
param([ValidateSet("Enable", "Disable", "Status")][string]$Action = "Status")
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_env.ps1"
Require-Venv
$Group = "KassaSentinel-NoEgress"
function Test-Admin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Get-Programs {
    $base = (& $Py -c "import sys; print(sys._base_executable)" | Out-String).Trim()
    @($Py, $base) | Where-Object { $_ -and (Test-Path $_) } | ForEach-Object { (Resolve-Path $_).Path } | Select-Object -Unique
}
$Remote = @("0.0.0.0-126.255.255.255", "128.0.0.0-255.255.255.255", "::2-ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")

if ($Action -ne "Status" -and -not (Test-Admin)) { Write-Host "Run PowerShell as Administrator for -Action $Action" -ForegroundColor Red; exit 1 }
switch ($Action) {
    "Enable" {
        Get-NetFirewallRule -Group $Group -ErrorAction SilentlyContinue | Remove-NetFirewallRule
        $i = 0
        foreach ($prog in Get-Programs) {
            $i++
            New-NetFirewallRule -DisplayName "$Group $i" -Group $Group -Direction Outbound -Action Block `
                -Program $prog -RemoteAddress $Remote -Profile Any -Enabled True | Out-Null
            Write-Host "BLOCK outbound (non-loopback): $prog"
        }
    }
    "Disable" {
        Get-NetFirewallRule -Group $Group -ErrorAction SilentlyContinue | Remove-NetFirewallRule
        Write-Host "Rules of group $Group removed"
    }
    "Status" {
        $rules = Get-NetFirewallRule -Group $Group -ErrorAction SilentlyContinue
        if (-not $rules) { Write-Host "No $Group rules (python.exe egress NOT blocked by firewall)"; exit 1 }
        foreach ($r in $rules) {
            $app = ($r | Get-NetFirewallApplicationFilter).Program
            Write-Host ("{0}: {1} {2} enabled={3} program={4}" -f $r.DisplayName, $r.Direction, $r.Action, $r.Enabled, $app)
        }
    }
}
