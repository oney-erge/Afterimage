# Sets up Afterimage on Windows the first time; every time after that,
# just starts it. Double-click run.bat, or run this file directly -- same
# thing.
#
# NOTE: this project was developed and its GPU decode kernels validated
# under WSL2, not native Windows CUDA -- Triton's native-Windows support is
# comparatively new and less battle-tested here than the WSL2 path. If
# `afterimage doctor` reports triton as unavailable or GPU tests fail,
# WSL2 + install.sh is the better-verified route on this platform.

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RepoDir ".venv"

function Log($msg) { Write-Host "[start] $msg" }

function Detect-Gpu {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try { nvidia-smi -L | Out-Null; return "nvidia" } catch {}
    }
    return "none"  # ROCm has no supported native-Windows PyTorch build today
}

# cu124 silently can't use an RTX 50-series (Blackwell) card, whose driver
# needs cu128+. $env:AFTERIMAGE_TORCH_INDEX_URL always overrides this.
function Get-CudaTorchIndex {
    if ($env:AFTERIMAGE_TORCH_INDEX_URL) { return $env:AFTERIMAGE_TORCH_INDEX_URL }
    try {
        $smi = & nvidia-smi 2>$null
        $match = ($smi | Select-String -Pattern 'CUDA Version:\s*([0-9]+)\.([0-9]+)').Matches
        if ($match.Count -gt 0) {
            $major = [int]$match[0].Groups[1].Value
            $minor = [int]$match[0].Groups[2].Value
            if ($major -gt 12 -or ($major -eq 12 -and $minor -ge 8)) {
                return "https://download.pytorch.org/whl/cu128"
            }
        }
    } catch {}
    return "https://download.pytorch.org/whl/cu124"
}

$AfterimageExe = Join-Path $VenvDir "Scripts\afterimage.exe"
$Reinstall = $args -contains "--reinstall"

if ((Test-Path $AfterimageExe) -and -not $Reinstall) {
    & $AfterimageExe serve
    exit $LASTEXITCODE
}

if ($Reinstall -and (Test-Path $VenvDir)) {
    Log "rebuilding from scratch"
    Remove-Item -Recurse -Force $VenvDir
}

Log "first run -- setting things up (this takes a few minutes)"
$GpuVendor = Detect-Gpu

python -m venv $VenvDir
& "$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip wheel | Out-Null

if ($GpuVendor -eq "nvidia") {
    $torchIndex = Get-CudaTorchIndex
    Log "installing CUDA torch build ($torchIndex)"
    pip install torch --index-url $torchIndex
    pip install -e "$RepoDir[gpu,server]"
} else {
    Log "no supported GPU detected -- installing CPU-only torch (inference will be slow;"
    Log "the fast GPU decode kernels in afterimage/runtime/gpu_decode*.py need CUDA,"
    Log "so this falls back to a compiled CPU decoder instead)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e "$RepoDir[server]"
}

try { afterimage doctor } catch { Log "doctor reported issues (see above)" }

Log "set up. Running a small model end to end first, so you can see it work"
Log "before waiting on a big download. Ctrl-C to skip straight to the server."
try { afterimage quickstart --yes } catch { Log "that didn't finish (see above), but the server will still start" }

Log "starting the server (Ctrl-C to stop; run this script again any time)"
afterimage serve
