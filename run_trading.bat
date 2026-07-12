@echo off
setlocal

cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
	echo Environnement virtuel introuvable. Lancez bootstrap_minimal.bat depuis C:\QuantProjects avant de demarrer le bot.
	exit /b 1
)

".venv\Scripts\python.exe" src\app\trading.py

endlocal
