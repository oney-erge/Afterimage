param(
    [string[]]$Models = @(
        "gemma3:4b",
        "qwen3-vl:4b-instruct",
        "qwen2.5:7b",
        "llama3.1:8b",
        "qwen3:8b",
        "qwen3-vl:8b-instruct",
        "gemma3:12b"
    ),
    [string]$OutFile = "c:\for fun\Afterimage\docs\ollama_capacity_results.json"
)

$ErrorActionPreference = "Continue"
$results = @()

function Get-VramUsedMiB {
    $line = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    return [int]($line.Trim())
}

function Wait-ForOllama {
    for ($i = 0; $i -lt 30; $i++) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
            return $true
        } catch { Start-Sleep -Milliseconds 500 }
    }
    return $false
}

if (-not (Wait-ForOllama)) { throw "ollama server not responding" }

$baselineVram = Get-VramUsedMiB
Write-Output "baseline VRAM (idle desktop): $baselineVram MiB"

$prompt = "In one sentence, what is the capital of France and its approximate population?"

foreach ($model in $Models) {
    Write-Output ""
    Write-Output "=================================================================="
    Write-Output "MODEL: $model"
    Write-Output "=================================================================="

    # unload anything currently resident so each model gets a clean measurement.
    # ollama stop writes spinner/ANSI control sequences to stderr even on
    # success, which PowerShell's strict native-command error handling turns
    # into a terminating NativeCommandError -- must be isolated in its own
    # try/catch with ErrorActionPreference relaxed, not just redirected.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { ollama stop $model *>$null } catch { }
    $ErrorActionPreference = $prevEap

    # A fixed 2s sleep was not enough: CUDA-level deallocation lags behind
    # ollama's own "stopped" bookkeeping, so nvidia-smi kept reading the
    # PREVIOUS model's memory as still resident when the next model's
    # "before" measurement was taken -- every delta in the first run of this
    # script was contaminated by whatever didn't get freed in time. Poll
    # until VRAM actually drops back near the pre-sweep baseline (or give up
    # after 15s and flag the measurement as unreliable) instead of assuming
    # a fixed delay is enough.
    $vramSettled = $false
    for ($i = 0; $i -lt 15; $i++) {
        $v = Get-VramUsedMiB
        if ($v -le ($baselineVram + 500)) { $vramSettled = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $vramSettled) {
        Write-Output "WARNING: VRAM did not settle back near baseline after unload -- delta for this model may be contaminated by a slow-releasing previous model"
    }

    $vramBefore = Get-VramUsedMiB

    $body = @{
        model = $model
        prompt = $prompt
        stream = $false
        options = @{ temperature = 0; num_predict = 80 }
    } | ConvertTo-Json

    $t0 = Get-Date
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
        $ok = $true
    } catch {
        Write-Output "GENERATION FAILED: $_"
        $ok = $false
        $resp = $null
    }
    $elapsed = ((Get-Date) - $t0).TotalSeconds

    $vramAfter = Get-VramUsedMiB
    $psInfo = ollama ps | Out-String

    if ($ok -and $resp) {
        $evalCount = $resp.eval_count
        $evalDurationNs = $resp.eval_duration
        $tps = if ($evalDurationNs -gt 0) { [math]::Round($evalCount / ($evalDurationNs / 1e9), 2) } else { $null }
        $answer = $resp.response.Trim()
    } else {
        $evalCount = $null; $tps = $null; $answer = $null
    }

    Write-Output "VRAM: $vramBefore -> $vramAfter MiB  (delta: $($vramAfter - $vramBefore) MiB)"
    Write-Output "tok/s: $tps   eval_count: $evalCount   wall: $([math]::Round($elapsed,1))s"
    Write-Output "ollama ps:"
    Write-Output $psInfo
    Write-Output "answer: $answer"

    $results += [PSCustomObject]@{
        model = $model
        success = $ok
        vram_before_mib = $vramBefore
        vram_after_mib = $vramAfter
        vram_delta_mib = $vramAfter - $vramBefore
        tok_per_s = $tps
        eval_count = $evalCount
        wall_seconds = [math]::Round($elapsed, 2)
        ollama_ps = $psInfo.Trim()
        answer = $answer
    }
}

$results | ConvertTo-Json -Depth 5 | Out-File -FilePath $OutFile -Encoding utf8
Write-Output ""
Write-Output "wrote $OutFile"
