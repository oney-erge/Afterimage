param(
  [ValidateSet("run", "doctor", "repair", "docker", "stop", "logs")]
  [string]$Action = "run",
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\scripts\install-utils.ps1
Initialize-Install -RepositoryRoot $PSScriptRoot -ProductName "Afterimage"
trap { Write-InstallFailure $_; Exit-InstallLock; exit 1 }
$UvVersion = "0.12.5"
$url = "http://127.0.0.1:8420"
function Resolve-Uv {
  $command = Get-Command uv -ErrorAction SilentlyContinue
  foreach ($candidate in @($(if ($command) { $command.Source }), "$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
  }
  return $null
}
function Write-Step {
  param([string]$Message)
  Write-Host "[Afterimage] $Message" -ForegroundColor Cyan
}
function Get-Uv {
  $uv = Resolve-Uv
  if ($uv) { Write-Step "Using uv at $uv"; return $uv }
  Write-Step "Installing uv $UvVersion..."
  $installer = Join-Path $env:TEMP "afterimage-uv-$UvVersion.ps1"
  try {
    Save-InstallDownload -Url "https://astral.sh/uv/$UvVersion/install.ps1" -Destination $installer -Label "uv download"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer
  } finally { Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue }
  $uv = Resolve-Uv
  if (-not $uv) { throw "uv installed but could not be located." }
  return $uv
}
function Wait-Ready {
  for ($i = 0; $i -lt 120; $i++) {
    try { Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2 | Out-Null; return $true } catch { Start-Sleep -Milliseconds 500 }
  }
  return $false
}
function Test-Ready {
  try { Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if ($Action -in @("docker", "stop", "logs")) {
  $docker = Get-Command docker -ErrorAction SilentlyContinue
  $engineRunning = $false
  if ($docker) { docker info *> $null; $engineRunning = ($LASTEXITCODE -eq 0) }
  if ($Action -eq "stop" -and -not $engineRunning) {
    Write-Host "The native server runs in the foreground. Press Ctrl+C in its terminal to stop it."
    exit 0
  }
  if ($Action -eq "logs" -and -not $engineRunning) {
    Write-Host "The native server writes logs to its foreground terminal."
    exit 0
  }
  if (-not $docker) { throw "Docker is not installed." }
  if (-not $engineRunning) { throw "Docker is installed but its engine is not running." }
  if ($Action -eq "stop") { docker compose down; exit $LASTEXITCODE }
  if ($Action -eq "logs") { docker compose logs --follow; exit $LASTEXITCODE }
  Enter-InstallLock
  Assert-InstallFreeSpace -Path $PSScriptRoot -RequiredGB 6
  docker compose up --detach --build
  if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed to start Afterimage." }
  if (-not (Wait-Ready)) { docker compose logs; throw "Afterimage did not become ready at $url." }
  Complete-Install
  Write-Host "Afterimage is ready at $url" -ForegroundColor Green
  if (-not $NoBrowser) { Start-Process $url }
  exit 0
}

$python = ".\.venv\Scripts\python.exe"
$exe = ".\.venv\Scripts\afterimage.exe"
if ($Action -eq "doctor") {
  if (-not (Test-Path -LiteralPath $exe)) { throw "Afterimage is not installed. Run .\run.bat once." }
  & $exe doctor; exit $LASTEXITCODE
}
Enter-InstallLock
Write-Step "Checking available disk space..."
Assert-InstallFreeSpace -Path $PSScriptRoot -RequiredGB 6
$uv = Get-Uv
Write-Step "Ensuring Python 3.11 is available..."
if (Test-Path -LiteralPath $python) {
  Write-Host "The project Python environment is already installed."
} else {
  Invoke-InstallRetry "Python installation" {
    $output = & $uv python install 3.11 2>&1
    if ($LASTEXITCODE -ne 0) { throw "uv python install failed: $($output -join [Environment]::NewLine)" }
    $output | Write-Host
  }
}
if (-not (Test-Path -LiteralPath $python)) {
  Write-Step "Creating the Afterimage virtual environment..."
  & $uv venv --python 3.11 .venv
  if ($LASTEXITCODE -ne 0) { throw "uv venv exited with $LASTEXITCODE" }
}
$gpu = "cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi -L *> $null
  if ($LASTEXITCODE -eq 0) { $gpu = "nvidia" }
}
$nativeWindows = [Environment]::OSVersion.Platform -eq "Win32NT"
$platformTag = if ($nativeWindows) { "windows" } else { "unix" }
$fingerprint = "$((Get-FileHash pyproject.toml -Algorithm SHA256).Hash)|uv=$UvVersion|python=3.11|$gpu|$platformTag"
$marker = ".\.venv\.afterimage-sync"
$installed = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -Raw).Trim() } else { "" }
if ($Action -eq "repair" -or $installed -ne $fingerprint -or -not (Test-Path -LiteralPath $exe)) {
  $reinstall = @()
  if ($Action -eq "repair") { $reinstall = @("--reinstall") }
  $torchIndex = if ($gpu -eq "nvidia") { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
  Invoke-InstallRetry "PyTorch installation" {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & $uv pip install --python $python @reinstall torch --index-url $torchIndex 2>&1
    $installExit = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    $output | Write-Host
    if ($installExit -ne 0) { throw "PyTorch installation failed: $($output -join [Environment]::NewLine)" }
  }
  $extras = if ($gpu -eq "nvidia") { ".[gpu,server]" } else { ".[server]" }
  if ($nativeWindows -and $gpu -eq "nvidia") {
    $extras = ".\[server]"
    Write-Host "Native Windows CUDA detected. Installing Windows-compatible server dependencies; Triton GPU kernels are Linux/WSL2-only." -ForegroundColor Yellow
  }
  Write-Step "Installing Afterimage and server dependencies..."
  Invoke-InstallRetry "Afterimage installation" {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & $uv pip install --python $python @reinstall --editable $extras 2>&1
    $installExit = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    $output | Write-Host
    if ($installExit -ne 0) { throw "Afterimage installation failed: $($output -join [Environment]::NewLine)" }
  }
  Set-Content -LiteralPath $marker -Value $fingerprint -NoNewline
} else {
  Write-Step "Dependencies are already up to date."
}
Write-Step "Running hardware and dependency checks..."
& $exe doctor
if ($LASTEXITCODE -ne 0) { throw "Afterimage doctor failed with exit $LASTEXITCODE." }
Complete-Install
if (Test-Ready) {
  Write-Host "Afterimage is already running at $url" -ForegroundColor Green
  if (-not $NoBrowser) { Start-Process $url }
  exit 0
}
Write-Step "Starting the local web UI. This terminal stays attached to the server..."
$serveArgs = @("serve")
if (-not $NoBrowser) { $serveArgs += "--open" }
& $exe @serveArgs
exit $LASTEXITCODE
