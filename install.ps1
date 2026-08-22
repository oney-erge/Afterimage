# Afterimage installer for native Windows: detects GPU vendor, installs the
# matching torch build, creates a venv, editable-installs the package, and
# runs the hardware diagnosis. Re-running when already installed just
# launches the server instead of reinstalling.
#
# NOTE: this project was developed and its GPU decode kernels validated
# under WSL2, not native Windows CUDA -- Triton's native-Windows support is
# comparatively new and less battle-tested here than the WSL2 path. If
# `afterimage doctor` reports triton as unavailable or GPU tests fail,
# WSL2 + install.sh is the better-verified route on this platform.

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RepoDir ".venv"

function Log($msg) { Write-Host "[install] $msg" }

function Detect-Gpu {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try { nvidia-smi -L | Out-Null; return "nvidia" } catch {}
    }
    return "none"  # ROCm has no supported native-Windows PyTorch build today
}

$AfterimageExe = Join-Path $VenvDir "Scripts\afterimage.exe"
$Reinstall = $args -contains "--reinstall"

if ((Test-Path $AfterimageExe) -and -not $Reinstall) {
    Log "already installed at $VenvDir -- launching (pass -- --reinstall to rebuild)"
    & $AfterimageExe serve
    exit $LASTEXITCODE
}

if ($Reinstall -and (Test-Path $VenvDir)) {
    Log "removing existing venv for a clean reinstall"
    Remove-Item -Recurse -Force $VenvDir
}

Log "setting up a new environment at $VenvDir"
$GpuVendor = Detect-Gpu
Log "detected GPU vendor: $GpuVendor"

python -m venv $VenvDir
& "$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip wheel | Out-Null

if ($GpuVendor -eq "nvidia") {
    Log "installing CUDA torch build"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -e "$RepoDir[gpu,server]"
} else {
    Log "no supported GPU detected -- installing CPU-only torch (inference will be slow;"
    Log "the GPU decode kernels in afterimage/runtime/gpu_decode*.py require CUDA)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e "$RepoDir[server]"
}

Log "running hardware diagnosis"
try { afterimage doctor } catch { Log "doctor reported issues (see above)" }

Log "install complete."
Log "Running the quickstart (compresses a small model, ~2 GB, and generates a"
Log "few tokens) to prove this install actually works before you commit an"
Log "hour and ~50 GB to a real model. Ctrl-C to skip it."
try { afterimage quickstart --yes } catch { Log "quickstart did not finish (see above)" }

Log "Launching the server (Ctrl-C to stop; re-run this script any time to relaunch)."
afterimage serve
