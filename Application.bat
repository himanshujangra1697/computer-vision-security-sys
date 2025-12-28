@echo off
TITLE Surveillance System Launcher
echo Starting Surveillance System...
echo ==========================================

:: 1. Navigate to the script's directory (ensures paths are correct)
cd /d "%~dp0"

:: 2. Activate the Virtual Environment
:: (If your venv folder is named differently, change 'venv' below)
call venv\Scripts\activate.bat

:: 3. Run the Python Script
:: We move into 'src/processor' first so imports work correctly
cd src\processor
python main.py

:: 4. Pause so you can see errors if it crashes
echo.
echo System stopped.
pause