@echo off
REM USD/JPY TFT 日次予測タスクのスケジューラ登録
REM 毎営業日 07:00 JST に predict.py を実行する
REM
REM 使い方: 管理者権限でこのバッチファイルを実行
REM   scripts\schedule_task.bat

set TASK_NAME=FX-Speculate-Daily-Predict
set PROJECT_DIR=%~dp0..
set PYTHON_CMD=uv run python scripts/predict.py

echo タスク名: %TASK_NAME%
echo プロジェクト: %PROJECT_DIR%
echo コマンド: %PYTHON_CMD%
echo.

schtasks /create /tn "%TASK_NAME%" /tr "cmd /c cd /d \"%PROJECT_DIR%\" && %PYTHON_CMD%"  /sc daily /st 07:00 /f

if %ERRORLEVEL% == 0 (
    echo.
    echo タスクが正常に登録されました。
    echo 確認: schtasks /query /tn "%TASK_NAME%"
) else (
    echo.
    echo エラー: タスクの登録に失敗しました。管理者権限で実行してください。
)

pause
