@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d c:\QuantProjects\ib_trading_bot

set LOG_DIR=logs\tick_ofi
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format yyyyMMdd"') do set TODAY=%%a
set LOG_FILE=%LOG_DIR%\collect_log_%TODAY%.txt

if not exist %LOG_DIR% mkdir %LOG_DIR%

echo [%date% %time%] Demarrage collecte tick OFI >> %LOG_FILE%

.venv\Scripts\python.exe src/app/collect_tick_ofi.py ^
    --host 127.0.0.1 ^
    --port 7496 ^
    --exchange ISLAND ^
    --symbols AAPL,MSFT,NVDA,AMZN,META ^
    --duration-min 395 ^
    --sample-sec 5 ^
    --client 98 ^
    >> %LOG_FILE% 2>&1

echo [%date% %time%] Collecte terminee (exit code: %ERRORLEVEL%) >> %LOG_FILE%
