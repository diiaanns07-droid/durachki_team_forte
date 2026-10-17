# Kassa Sentinel - stop OUR service only (owner: A17). ASCII-only.
# Kills only PIDs recorded in runs\deploy\state\state.json whose start time matches. Data in data\ is kept.
. "$PSScriptRoot\_env.ps1"
Require-Venv
Set-Location $Repo
& $Py deploy\supervisor.py stop
exit $LASTEXITCODE
