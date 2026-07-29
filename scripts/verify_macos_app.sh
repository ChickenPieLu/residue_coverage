#!/bin/zsh
set -euo pipefail

PROJECT_ROOT=${0:A:h:h}
APP_PATH=${1:-"$PROJECT_ROOT/dist/ResidueCoverage.app"}
MODEL_NAME="smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
MODEL_PATH="$APP_PATH/Contents/Resources/models/$MODEL_NAME"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/ResidueCoverage"

if [[ ! -d "$APP_PATH" ]]; then
  print -u2 "错误：应用不存在：$APP_PATH"
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  print -u2 "错误：应用内模型不存在：$MODEL_PATH"
  exit 1
fi
if [[ ! -x "$EXECUTABLE_PATH" ]]; then
  print -u2 "错误：应用主程序不存在或不可执行：$EXECUTABLE_PATH"
  exit 1
fi

print "主程序架构：$(lipo -archs "$EXECUTABLE_PATH")"
if [[ "$(lipo -archs "$EXECUTABLE_PATH")" != "arm64" ]]; then
  print -u2 "错误：主程序不是纯 arm64。"
  exit 1
fi

EXPECTED_MODEL_SHA=$(awk '{print $1}' "$PROJECT_ROOT/MODEL_APP_CHECKSUM.sha256")
ACTUAL_MODEL_SHA=$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')
if [[ "$EXPECTED_MODEL_SHA" != "$ACTUAL_MODEL_SHA" ]]; then
  print -u2 "错误：应用内模型 SHA-256 不匹配。"
  exit 1
fi
print "模型 SHA-256：$ACTUAL_MODEL_SHA"

UNEXPECTED=$(find "$APP_PATH" \
  \( -name ".venv" \
  -o -name ".git" \
  -o -name "residue_background" \
  -o -name "training" \
  -o -name "legacy" \
  -o -name "logs" \
  -o -name "*.pth" ! -name "$MODEL_NAME" \) \
  -print)
if [[ -n "$UNEXPECTED" ]]; then
  print -u2 "错误：应用包含不应交付的内容："
  print -u2 "$UNEXPECTED"
  exit 1
fi

NON_ARM64=""
while IFS= read -r -d $'\0' candidate; do
  description=$(file -b "$candidate")
  if [[ "$description" == *"Mach-O"* && "$description" != *"arm64"* ]]; then
    NON_ARM64+="$candidate: $description"$'\n'
  fi
done < <(find "$APP_PATH" -type f -print0)
if [[ -n "$NON_ARM64" ]]; then
  print -u2 "错误：发现非 arm64 Mach-O 文件："
  print -u2 "$NON_ARM64"
  exit 1
fi

codesign --verify --deep --strict --verbose=2 "$APP_PATH"
print "应用包内容、arm64 架构、模型校验和及签名结构检查通过。"
