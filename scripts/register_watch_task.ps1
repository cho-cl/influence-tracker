<#
.SYNOPSIS
Creates (or updates) the scheduled task that keeps `influence watch` running for live phone alerts, or removes it.

.DESCRIPTION
Starts at logon and every day at 03:50 local time, restarts within a minute if it stops, and never runs two copies.
The watch keeps the PC awake from 04:00 to 20:00 New York time on trading days; a closed laptop lid can still
sleep the machine depending on Windows power settings.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\register_watch_task.ps1
powershell -ExecutionPolicy Bypass -File scripts\register_watch_task.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string]$TaskName = "InfluenceTracker Watch",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_watch.cmd"

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName' exists; nothing to remove."
    }
    return
}

if (-not (Test-Path $runner)) { throw "Cannot find $runner." }
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$argument = '/c ""{0}""' -f $runner
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $argument -WorkingDirectory $repo
$at = [datetime]::ParseExact("03:50", [string[]]@("HH:mm"), [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::None)
$daily = New-ScheduledTaskTrigger -Daily -At $at
# New-ScheduledTaskTrigger pins a UTC offset; without one, Task Scheduler follows local daylight saving.
$daily.StartBoundary = $at.ToString("yyyy-MM-dd'T'HH:mm:ss", [Globalization.CultureInfo]::InvariantCulture)
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $user), $daily)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$description = "influence-tracker live alerts: checks for stock posts every few minutes and notifies phones. " +
    "Output: $repo\logs\watch.log"

$null = Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
    -Principal $principal -Description $description -Force

Write-Host "Registered scheduled task '$TaskName' (at logon + daily 03:50, restarts on failure)."
Write-Host "Start it now:   Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "Remove it:      powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Unregister"
