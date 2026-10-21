# Checks the launcher helper in scripts/install-utils.ps1 under the PowerShell that run.bat starts
# (Windows PowerShell 5.1). Run it with:  powershell -NoProfile -File scripts/test-launcher-helpers.ps1
# A plain progress line on stderr from uv once aborted every first run under 5.1, and CI only parsed the launcher.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "install-utils.ps1")

function Assert-That([bool]$Condition, [string]$Message) {
  if (-not $Condition) { throw "FAILED: $Message" }
  Write-Host "ok: $Message"
}

$result = Invoke-InstallNative "cmd.exe" @("/c", "echo progress on stderr 1>&2 & exit 3")
Assert-That ($result.ExitCode -eq 3) "native exit code is returned (got $($result.ExitCode))"
Assert-That (($result.Output -join " ") -match "progress on stderr") "stderr text is captured as a plain line"
Assert-That ($ErrorActionPreference -eq "Stop") "the caller's ErrorActionPreference is restored"

$clean = Invoke-InstallNative "cmd.exe" @("/c", "echo fine & exit 0")
Assert-That ($clean.ExitCode -eq 0) "a successful command returns exit code 0"
Write-Host "All launcher helper checks passed."
