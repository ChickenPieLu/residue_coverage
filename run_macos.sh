#!/bin/bash
set -u

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
MODEL_NAME="smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
MODEL_PATH="$PROJECT_ROOT/$MODEL_NAME"
CHECKSUM_FILE="$PROJECT_ROOT/MODEL_CHECKSUMS.sha256"
MODEL_SHA256="$(awk -v name="$MODEL_NAME" '$2 == name {print $1}' "$CHECKSUM_FILE")"

cd "$PROJECT_ROOT"

if [[ -z "$MODEL_SHA256" ]]; then
  printf '错误：%s 中缺少默认模型的 SHA-256。\n' "$CHECKSUM_FILE" >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  printf '错误：尚未安装运行环境。\n' >&2
  printf '请先运行 ./setup_macos.sh，然后再启动应用。\n' >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  printf '错误：未找到模型文件：\n%s\n' "$MODEL_PATH" >&2
  printf '请先运行 ./setup_macos.sh，或把 %s 放到项目根目录。\n' \
    "$MODEL_NAME" >&2
  exit 1
fi

ACTUAL_SHA256="$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$MODEL_SHA256" ]]; then
  printf '错误：模型文件校验失败，文件可能不完整或已损坏。\n' >&2
  printf '请重新获取模型后再运行。期望 SHA-256：%s\n' "$MODEL_SHA256" >&2
  exit 1
fi

printf '正在启动 ResidueCoverage，本地网页将由默认浏览器打开…\n'
if "$VENV_PYTHON" "$PROJECT_ROOT/app.py"; then
  exit 0
else
  STATUS=$?
  printf '\nResidueCoverage 启动失败（错误代码 %s）。\n' "$STATUS" >&2
  printf '请重新运行 ./setup_macos.sh；若仍失败，请保存上方信息联系项目维护者。\n' >&2
  exit "$STATUS"
fi
