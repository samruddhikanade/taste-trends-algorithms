# ---------------------------------------------------------------------------
# Download data -> run the Spark pipeline -> run tests -> build the ZIP.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 -Mode dev
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 -Mode full
# Optional: -Sample 0.25 to use 25% of users from MovieLens 25M.
# ---------------------------------------------------------------------------
param(
    [ValidateSet("dev", "full")][string]$Mode = "dev",
    [double]$Sample = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts\windows\setup.ps1 first." }

function Step($title, [string[]]$arguments) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $py @arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $title" }
}

Step "Download MovieLens ($Mode)" @("scripts\download_data.py", "--mode", $Mode)
$pipelineArgs = @("scripts\run_pipeline.py", "--mode", $Mode)
if ($Sample -gt 0) { $pipelineArgs += @("--sample", "$Sample") }
Step "Spark pipeline ($Mode)" $pipelineArgs
Step "Automated tests" @("-m", "pytest")
Step "Build and verify ZIP" @("scripts\package_zip.py")

Write-Host "`nAll done. Launch the dashboard with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\streamlit.exe run app.py"
