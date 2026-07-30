@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "MODEL_NAME=smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
set "MODEL_PATH=%PROJECT_ROOT%\%MODEL_NAME%"
set "MODEL_TEMP=%MODEL_PATH%.part"
set "CHECKSUM_FILE=%PROJECT_ROOT%\MODEL_CHECKSUMS.sha256"
set "MODEL_SHA256="
set "MODEL_DOWNLOAD_URL=https://github.com/ChickenPieLu/residue_coverage/releases/latest/download/smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
set "PYTHON_COMMAND="

for /f "usebackq tokens=1,2" %%A in ("%CHECKSUM_FILE%") do if "%%B"=="%MODEL_NAME%" set "MODEL_SHA256=%%A"
if not defined MODEL_SHA256 (
  echo 错误：MODEL_CHECKSUMS.sha256 中缺少默认模型的 SHA-256。
  exit /b 1
)

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHON_COMMAND=py -3.12"

if not defined PYTHON_COMMAND (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_COMMAND=python"
)

if not defined PYTHON_COMMAND (
  echo 错误：未找到 Python 3.12。
  echo 请先从 https://www.python.org/downloads/ 安装 Python 3.12，
  echo 并在安装界面勾选 “Add python.exe to PATH”。
  exit /b 1
)

if not exist "%VENV_PYTHON%" (
  echo 正在创建项目虚拟环境 .venv...
  %PYTHON_COMMAND% -m venv "%PROJECT_ROOT%\.venv"
  if errorlevel 1 (
    echo 错误：无法创建虚拟环境，请确认 Python 3.12 安装完整。
    exit /b 1
  )
) else (
  "%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
  if errorlevel 1 (
    echo 错误：现有 .venv 不是 Python 3.12 环境。
    echo 请先将 .venv 移到其他位置，再重新运行此脚本。
    exit /b 1
  )
)

call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
if errorlevel 1 (
  echo 错误：无法激活虚拟环境。
  exit /b 1
)

echo 正在升级 pip...
python -m pip install --upgrade pip
if errorlevel 1 (
  echo 错误：pip 升级失败，请检查网络连接。
  exit /b 1
)

echo 正在安装运行依赖（PyTorch 下载较大，第一次可能需要较长时间）...
python -m pip install -r "%PROJECT_ROOT%\requirements-runtime.txt"
if errorlevel 1 (
  echo 错误：依赖安装失败，请检查网络后重新运行此脚本。
  exit /b 1
)

if not exist "%MODEL_PATH%" (
  if not defined MODEL_DOWNLOAD_URL (
    echo.
    echo 运行环境已安装，但尚未找到模型文件：
    echo %MODEL_PATH%
    echo 请从项目发布者处单独获取 %MODEL_NAME%，放到项目根目录后重新运行此脚本。
    exit /b 2
  )
  echo 正在下载默认模型（约 93 MB）...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%MODEL_DOWNLOAD_URL%' -OutFile '%MODEL_TEMP%'"
  if errorlevel 1 (
    if exist "%MODEL_TEMP%" del /q "%MODEL_TEMP%"
    echo 错误：模型下载失败，请检查网络后重试。
    exit /b 1
  )
  call :verify_model "%MODEL_TEMP%"
  if errorlevel 1 (
    if exist "%MODEL_TEMP%" del /q "%MODEL_TEMP%"
    echo 错误：下载的模型未通过校验，临时文件已删除。
    exit /b 1
  )
  move /y "%MODEL_TEMP%" "%MODEL_PATH%" >nul
  if errorlevel 1 (
    echo 错误：无法把已校验模型移动到项目根目录。
    exit /b 1
  )
) else (
  call :verify_model "%MODEL_PATH%"
  if errorlevel 1 exit /b 1
)

echo.
echo 安装完成。以后双击 run_windows.bat 即可启动网页界面。
exit /b 0

:verify_model
set "VERIFY_TARGET=%~1"
set "ACTUAL_SHA256="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%VERIFY_TARGET%').Hash.ToLowerInvariant()"`) do set "ACTUAL_SHA256=%%H"
if not defined ACTUAL_SHA256 (
  echo 错误：无法计算模型文件 SHA-256。
  exit /b 1
)
if /I not "%ACTUAL_SHA256%"=="%MODEL_SHA256%" (
  echo 错误：模型文件 SHA-256 校验失败。
  echo 期望：%MODEL_SHA256%
  echo 实际：%ACTUAL_SHA256%
  echo 请删除损坏的模型文件后重新运行此脚本。
  exit /b 1
)
echo 模型校验通过：%ACTUAL_SHA256%
exit /b 0
