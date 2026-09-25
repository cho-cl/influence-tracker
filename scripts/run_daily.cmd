@echo off
rem Runs `influence daily` for Windows Task Scheduler. Output is appended to logs\scheduled.log.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist logs mkdir logs
>>logs\scheduled.log echo ===== %DATE% %TIME% influence daily =====
".venv\Scripts\influence.exe" daily >>logs\scheduled.log 2>&1
set RC=%ERRORLEVEL%
>>logs\scheduled.log echo ===== %DATE% %TIME% exit %RC% =====
exit /b %RC%
