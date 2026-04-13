@echo off
REM USD/JPY TFT 日次予測 - タスクスケジューラ用
REM 毎朝 7:00 に実行される想定

cd /d C:\Users\daiya\Documents\FX-speculate

set PYTHONPATH=.

if not exist logs mkdir logs

echo [%date% %time%] Starting daily prediction... >> logs\daily_predict.log
uv run python scripts/predict.py >> logs\daily_predict.log 2>&1
echo [%date% %time%] Done (exit code: %ERRORLEVEL%) >> logs\daily_predict.log
