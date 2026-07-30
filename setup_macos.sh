#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
MODEL_NAME="smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
MODEL_PATH="$PROJECT_ROOT/$MODEL_NAME"
MODEL_TEMP="$PROJECT_ROOT/$MODEL_NAME.part"
CHECKSUM_FILE="$PROJECT_ROOT/MODEL_CHECKSUMS.sha256"
MODEL_SHA256="$(awk -v name="$MODEL_NAME" '$2 == name {print $1}' "$CHECKSUM_FILE")"
MODEL_DOWNLOAD_URL=""
PYTHON_COMMAND="${RESIDUE_COVERAGE_PYTHON:-python3}"

if [[ -z "$MODEL_SHA256" ]]; then
  printf '错误：%s 中缺少默认模型的 SHA-256。\n' "$CHECKSUM_FILE" >&2
  exit 1
fi

verify_model() {
  local actual_sha256
  actual_sha256="$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$MODEL_SHA256" ]]; then
    printf '错误：模型文件 SHA-256 校验失败。\n' >&2
    printf '期望：%s\n实际：%s\n' "$MODEL_SHA256" "$actual_sha256" >&2
    printf '请删除损坏的模型文件后重新运行此脚本。\n' >&2
    return 1
  fi
  printf '模型校验通过：%s\n' "$actual_sha256"
}

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
  printf '错误：未找到 Python 3.12。\n' >&2
  printf '请先从 https://www.python.org/downloads/ 安装 Python 3.12。\n' >&2
  exit 1
fi

if ! "$PYTHON_COMMAND" -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'
then
  printf '错误：需要 Python 3.12，当前命令是：' >&2
  "$PYTHON_COMMAND" --version >&2
  printf '可设置 RESIDUE_COVERAGE_PYTHON 指向 Python 3.12。\n' >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  printf '正在创建项目虚拟环境 .venv…\n'
  "$PYTHON_COMMAND" -m venv "$VENV_DIR"
elif ! "$VENV_PYTHON" -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'
then
  printf '错误：现有 .venv 不是 Python 3.12 环境。\n' >&2
  printf '请先将 .venv 移到其他位置，再重新运行此脚本。\n' >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
printf '正在升级 pip…\n'
python -m pip install --upgrade pip
printf '正在安装运行依赖（PyTorch 下载较大，第一次可能需要较长时间）…\n'
python -m pip install -r "$PROJECT_ROOT/requirements-runtime.txt"

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ -z "$MODEL_DOWNLOAD_URL" ]]; then
    printf '\n运行环境已安装，但尚未找到模型文件：\n%s\n' "$MODEL_PATH" >&2
    printf '请从项目发布者处单独获取 %s，放到项目根目录后重新运行此脚本。\n' \
      "$MODEL_NAME" >&2
    exit 2
  fi
  printf '正在下载默认模型（约 93 MB）…\n'
  if ! curl -fL --progress-bar "$MODEL_DOWNLOAD_URL" -o "$MODEL_TEMP"; then
    rm -f "$MODEL_TEMP"
    printf '错误：模型下载失败，请检查网络后重试。\n' >&2
    exit 1
  fi
  mv "$MODEL_TEMP" "$MODEL_PATH"
fi

verify_model
printf '\n安装完成。以后双击或运行 ./run_macos.sh 即可启动网页界面。\n'
