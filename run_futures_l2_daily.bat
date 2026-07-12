@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d c:\QuantProjects\ib_trading_bot

set LOG_DIR=logs\futures_l2
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format yyyyMMdd"') do set TODAY=%%a
set LOG_FILE=%LOG_DIR%\collect_log_%TODAY%.txt

if not exist %LOG_DIR% mkdir %LOG_DIR%

echo [%date% %time%] Demarrage collecte Futures L2 >> %LOG_FILE%

.venv\Scripts\python.exe src/app/collect_futures_l2.py ^
    --host 192.168.0.103 ^
    --port 7496 ^
    --client 97 ^
    --symbols MES,MNQ,YM ^
    --no-ticks ^
    --duration-min 480 ^
    --output-dir collectOF ^
    >> %LOG_FILE% 2>&1

echo [%date% %time%] Collecte terminee (exit code: %ERRORLEVEL%) >> %LOG_FILE%
