@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "MODEL_NAME=smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
set "MODEL_PATH=%PROJECT_ROOT%\%MODEL_NAME%"
set "CHECKSUM_FILE=%PROJECT_ROOT%\MODEL_CHECKSUMS.sha256"
set "MODEL_SHA256="

for /f "usebackq tokens=1,2" %%A in ("%CHECKSUM_FILE%") do if "%%B"=="%MODEL_NAME%" set "MODEL_SHA256=%%A"
if not defined MODEL_SHA256 (
  echo 错误：MODEL_CHECKSUMS.sha256 中缺少默认模型的 SHA-256。
  exit /b 1
)

if not exist "%VENV_PYTHON%" (
  echo 错误：尚未安装运行环境。
  echo 请先双击 setup_windows.bat，然后再启动应用。
  exit /b 1
)

if not exist "%MODEL_PATH%" (
  echo 错误：未找到模型文件：
  echo %MODEL_PATH%
  echo 请先运行 setup_windows.bat，或把 %MODEL_NAME% 放到项目根目录。
  exit /b 1
)

set "ACTUAL_SHA256="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%MODEL_PATH%').Hash.ToLowerInvariant()"`) do set "ACTUAL_SHA256=%%H"
if not defined ACTUAL_SHA256 (
  echo 错误：无法计算模型文件 SHA-256。
  exit /b 1
)
if /I not "%ACTUAL_SHA256%"=="%MODEL_SHA256%" (
  echo 错误：模型文件校验失败，文件可能不完整或已损坏。
  echo 请重新获取模型后再运行。期望 SHA-256：%MODEL_SHA256%
  exit /b 1
)

echo 正在启动 ResidueCoverage，本地网页将由默认浏览器打开...
"%VENV_PYTHON%" "%PROJECT_ROOT%\app.py"
set "APP_STATUS=%ERRORLEVEL%"
if not "%APP_STATUS%"=="0" (
  echo.
  echo ResidueCoverage 启动失败（错误代码 %APP_STATUS%）。
  echo 请重新运行 setup_windows.bat；若仍失败，请保存上方信息联系项目维护者。
  exit /b %APP_STATUS%
)

exit /b 0
