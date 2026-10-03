@echo off
rem Runs `influence watch` for Windows Task Scheduler. Warnings and errors are appended to logs\watch.log;
rem the full log is logs\influence-watch.log.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist logs mkdir logs
>>logs\watch.log echo ===== %DATE% %TIME% influence watch =====
".venv\Scripts\influence.exe" watch --quiet >>logs\watch.log 2>&1
set RC=%ERRORLEVEL%
>>logs\watch.log echo ===== %DATE% %TIME% exit %RC% =====
exit /b %RC%
