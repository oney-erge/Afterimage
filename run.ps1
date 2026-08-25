param(
  [ValidateSet("run", "doctor", "repair", "docker", "stop", "logs")]
  [string]$Action = "run",
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$UvVersion = "0.12.5"
$url = "http://127.0.0.1:8420"

function Invoke-Retry([string]$Label, [scriptblock]$Operation) {
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    try { & $Operation; return } catch {
      if ($attempt -eq 3) { throw "$Label failed after 3 attempts: $($_.Exception.Message)" }
      Start-Sleep -Seconds ([math]::Pow(2, $attempt - 1))
    }
  }
}
function Resolve-Uv {
  $command = Get-Command uv -ErrorAction SilentlyContinue
  foreach ($candidate in @($(if ($command) { $command.Source }), "$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
  }
  return $null
}
function Ensure-Uv {
  $uv = Resolve-Uv
  if ($uv) { return $uv }
  $installer = Join-Path $env:TEMP "afterimage-uv-$UvVersion.ps1"
  try {
    Invoke-Retry "uv download" { Invoke-WebRequest -UseBasicParsing -Uri "https://astral.sh/uv/$UvVersion/install.ps1" -OutFile $installer }
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
  docker compose up --detach --build
  if (-not (Wait-Ready)) { docker compose logs; throw "Afterimage did not become ready at $url." }
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
$uv = Ensure-Uv
Invoke-Retry "Python installation" { & $uv python install 3.11; if ($LASTEXITCODE -ne 0) { throw "uv python install exited with $LASTEXITCODE" } }
if (-not (Test-Path -LiteralPath $python)) {
  & $uv venv --python 3.11 .venv
  if ($LASTEXITCODE -ne 0) { throw "uv venv exited with $LASTEXITCODE" }
}
$gpu = "cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi -L *> $null
  if ($LASTEXITCODE -eq 0) { $gpu = "nvidia" }
}
$fingerprint = "$((Get-FileHash pyproject.toml -Algorithm SHA256).Hash)|uv=$UvVersion|python=3.11|$gpu"
$marker = ".\.venv\.afterimage-sync"
$installed = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -Raw).Trim() } else { "" }
if ($Action -eq "repair" -or $installed -ne $fingerprint -or -not (Test-Path -LiteralPath $exe)) {
  $reinstall = @()
  if ($Action -eq "repair") { $reinstall = @("--reinstall") }
  $torchIndex = if ($gpu -eq "nvidia") { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
  Invoke-Retry "PyTorch installation" { & $uv pip install --python $python @reinstall torch --index-url $torchIndex; if ($LASTEXITCODE -ne 0) { throw "uv pip exited with $LASTEXITCODE" } }
  $extras = if ($gpu -eq "nvidia") { ".[gpu,server]" } else { ".[server]" }
  Invoke-Retry "Afterimage installation" { & $uv pip install --python $python @reinstall --editable $extras; if ($LASTEXITCODE -ne 0) { throw "uv pip exited with $LASTEXITCODE" } }
  Set-Content -LiteralPath $marker -Value $fingerprint -NoNewline
}
& $exe doctor
$serveArgs = @("serve")
if (-not $NoBrowser) { $serveArgs += "--open" }
& $exe @serveArgs
exit $LASTEXITCODE
