<#
.SYNOPSIS
Creates (or updates) the scheduled task that keeps `influence watch` running for live phone alerts, or removes it.

.DESCRIPTION
Starts at logon and every day at 03:50 local time, and Task Scheduler tries again every 5 minutes, so a watch that
crashed or whose window was closed is back within 5 minutes. Never runs two copies. A console window stays open while
the watch runs: minimize it (closing it stops alerts for up to 5 minutes). Stop it for good with -Unregister.
The watch keeps the PC awake from 04:00 to 20:00 New York time on trading days; a closed laptop lid can still
sleep the machine depending on Windows power settings, and a sleeping PC sends nothing until it wakes.

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
$exe = Join-Path $repo ".venv\Scripts\influence.exe"
if (-not (Test-Path $exe)) {
    Write-Warning "Cannot find $exe. Run 'uv sync' in $repo first, or the task will fail every time it starts."
}
$envFile = Join-Path $repo ".env"
if (-not ((Test-Path $envFile) -and (Select-String -Path $envFile -Pattern '^\s*NTFY_TOPIC\s*=\s*\S' -Quiet))) {
    Write-Warning "NTFY_TOPIC is not set in .env. Run 'uv run influence alerts setup' first, or the watch refuses to start."
}
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$argument = '/c ""{0}""' -f $runner
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $argument -WorkingDirectory $repo
$at = [datetime]::ParseExact("03:50", [string[]]@("HH:mm"), [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::None)
$daily = New-ScheduledTaskTrigger -Daily -At $at
# New-ScheduledTaskTrigger pins a UTC offset; without one, Task Scheduler follows local daylight saving.
$daily.StartBoundary = $at.ToString("yyyy-MM-dd'T'HH:mm:ss", [Globalization.CultureInfo]::InvariantCulture)
# Task Scheduler's restart-on-failure only covers a task that fails to start, not a program that exits later. So the
# daily trigger repeats every 5 minutes all day; with IgnoreNew a repeat does nothing while the watch is running.
$repeat = New-ScheduledTaskTrigger -Once -At $at -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 1)
$daily.Repetition = $repeat.Repetition
$daily.Repetition.StopAtDurationEnd = $false  # True would stop the running watch at the end of each day
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $user), $daily)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$description = "influence-tracker live alerts: checks for stock posts every few minutes and notifies phones. " +
    "Output: $repo\logs\watch.log"

$null = Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
    -Principal $principal -Description $description -Force

Write-Host "Registered scheduled task '$TaskName' (at logon + daily 03:50, retried every 5 minutes if it stops)."
Write-Host "Start it now:   Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "Output:         $repo\logs\watch.log (problems) and $repo\logs\influence-watch.log (everything)"
Write-Host "Remove it:      powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Unregister"
