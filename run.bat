@echo off
echo Starting Face Recognition Attendance System...
echo 1. Classic UI
echo 2. Modern UI (with dark/light mode)
choice /c 12 /m "Select UI version"

if errorlevel 2 goto modern
if errorlevel 1 goto classic

:classic
python app.py
goto end

:modern
python app_modern.py
goto end

:end
pause 