@echo off
rem Run from the script's directory so relative paths work correctly
pushd "%~dp0"

rem Prefer the project's local virtual environment (try .venv then venv), otherwise fall back to system python
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "rlbc_daily_to_notion.py"
) else if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" "rlbc_daily_to_notion.py"
) else (
  echo Warning: no local virtualenv found, using system python
  python "rlbc_daily_to_notion.py"
)

popd
exit /b %ERRORLEVEL%
