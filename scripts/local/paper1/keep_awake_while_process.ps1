param(
    [Parameter(Mandatory = $true)]
    [int]$TargetProcessId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class Paper1PowerRequest
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint executionState);
}
'@

$continuous = [uint32]::Parse(
    '80000000',
    [Globalization.NumberStyles]::HexNumber
)
$systemRequired = [uint32]0x00000001

try {
    while (Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue) {
        $state = [Paper1PowerRequest]::SetThreadExecutionState(
            $continuous -bor $systemRequired
        )
        if ($state -eq 0) {
            throw 'SetThreadExecutionState failed.'
        }
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][Paper1PowerRequest]::SetThreadExecutionState($continuous)
}
