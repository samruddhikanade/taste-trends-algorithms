# ---------------------------------------------------------------------------
# One-time setup on Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
# Creates .venv, installs requirements and checks Java/Spark.
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
Write-Host "Project folder: $root" -ForegroundColor Cyan
if ($root -match " ") {
    Write-Warning "The path contains a space. Spark may fail to start. Move the folder to e.g. C:\BDA\ and rerun."
}

# Pick Python 3.11 or 3.10 through the Windows launcher, else fall back to 'python'.
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $ErrorActionPreference = "Continue"          # probing a missing version writes to stderr
    foreach ($v in @("3.11", "3.10")) {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
    }
    $ErrorActionPreference = "Stop"
}
if (-not $py) { $py = @("python") }
$pyArgs = @($py | Select-Object -Skip 1)
Write-Host "Using: $($py -join ' ')" -ForegroundColor Cyan

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $py[0] @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }

if (-not (Get-Command java -ErrorAction SilentlyContinue) -and -not $env:JAVA_HOME) {
    Write-Warning "Java was not found. Install it with:  winget install EclipseAdoptium.Temurin.17.JDK"
}
& $venvPy scripts\check_environment.py --spark
Write-Host "`nSetup finished. Next: powershell -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 -Mode dev" -ForegroundColor Green
