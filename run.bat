@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python не найден. Установи Python 3.10+ с https://www.python.org/downloads/ и поставь галочку "Add python.exe to PATH".
  pause & exit /b 1
)
if not exist .venv (
  py -m venv .venv || (pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt
echo Открой http://127.0.0.1:8000
python -m uvicorn gait.server:app --reload
pause
