<#
.SYNOPSIS
Creates (or updates) the daily Windows scheduled task that runs `influence daily`, or removes it.

.DESCRIPTION
The task runs scripts\run_daily.cmd with the repo as its working directory, for the current user,
only while that user is logged on (no password is stored). -At is local wall-clock time and stays
put across daylight-saving changes. The default, 20:30, assumes the machine is on US Eastern time:
extended-hours trading ends at 20:00 ET, and the price snapshot only saves sessions whose extended
hours are over, so this saves the same day's bars.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1 -At 21:15
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string]$At = "20:30",
    [string]$TaskName = "InfluenceTracker Daily",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

function ConvertTo-TaskTime([string]$Text) {
    # Without the [string[]] cast, PowerShell joins the formats into one "H:mm HH:mm" string that never matches.
    $formats = [string[]]@("H:mm", "HH:mm")
    try {
        return [datetime]::ParseExact($Text, $formats, [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::None)
    } catch {
        throw "-At must be a 24-hour local time like 20:30, got '$Text'."
    }
}

function New-LocalDailyTrigger([datetime]$Time) {
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time
    # New-ScheduledTaskTrigger pins a UTC offset; without one, Task Scheduler follows local daylight saving.
    $trigger.StartBoundary = $Time.ToString("yyyy-MM-dd'T'HH:mm:ss", [Globalization.CultureInfo]::InvariantCulture)
    return $trigger
}

# Dot-sourcing (as the tests do) only defines the functions above.
if ($MyInvocation.InvocationName -eq ".") { return }

$repo = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_daily.cmd"
$removeHint = "powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Unregister -TaskName `"$TaskName`""

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName' exists; nothing to remove."
    }
    return
}

$time = ConvertTo-TaskTime $At

if (-not (Test-Path $runner)) {
    throw "Cannot find $runner."
}
$exe = Join-Path $repo ".venv\Scripts\influence.exe"
if (-not (Test-Path $exe)) {
    Write-Warning "$exe does not exist yet. Run 'uv sync' in $repo before the task first fires."
}
$tz = (Get-TimeZone).Id
if ($tz -ne "Eastern Standard Time") {
    Write-Warning ("This machine's time zone is '$tz'. -At is local time; pick a time after 20:00 US Eastern " +
        "so each day's 1-minute bars are complete.")
}

$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
# cmd strips the outer pair of quotes, so the doubled pair keeps a path with spaces, & or ( intact.
$argument = '/c ""{0}""' -f $runner
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $argument -WorkingDirectory $repo
$trigger = New-LocalDailyTrigger $time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$description = "influence-tracker: collect posts (Truth Social, Reddit, ApeWisdom, X) and snapshot 1-minute bars. " +
    "Output: $repo\logs\scheduled.log"

$null = Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description $description -Force
$info = Get-ScheduledTaskInfo -TaskName $TaskName
$boundary = (Get-ScheduledTask -TaskName $TaskName).Triggers[0].StartBoundary
if ($boundary -match "(Z|[+-]\d\d:\d\d)$") {
    Write-Warning ("The registered start time '$boundary' has a fixed UTC offset, so the task will shift by an hour " +
        "when daylight saving changes. Untick 'Synchronize across time zones' on the trigger in Task Scheduler.")
}

Write-Host "Registered scheduled task '$TaskName':"
Write-Host "  runs:         cmd.exe $argument"
Write-Host "  working dir:  $repo"
Write-Host "  schedule:     daily at $($time.ToString('HH:mm')) local time ($tz), following daylight saving"
Write-Host "  user:         $user (only while logged on; no password stored)"
Write-Host "  settings:     start when available, runs on battery, 3 h time limit, no overlapping runs"
Write-Host "  next run:     $($info.NextRunTime)"
Write-Host "  output log:   $repo\logs\scheduled.log"
Write-Host ""
Write-Host "Run it now:     Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "Remove it:      $removeHint"
