@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_EXE=D:\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo 未找到虚拟环境 Python: %PYTHON_EXE%
  echo 请先确认 D:\.venv 是否存在，或修改本文件里的 PYTHON_EXE。
  pause
  exit /b 1
)

if not exist "results.csv" (
  echo 未找到 results.csv，先运行一次 CPU 实验...
  "%PYTHON_EXE%" train_federated.py --rounds 3 --samples-per-hospital 600 --mode research --cpu
)

echo 正在启动 Streamlit Demo...
echo 浏览器打开后，请访问 http://localhost:8501
"%PYTHON_EXE%" -m streamlit run demo\app.py

pause
