#!/bin/zsh
set -euo pipefail

PROJECT_ROOT=${0:A:h}
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
APP_NAME="ResidueCoverage"
MODEL_NAME="smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
DIST_APP="$PROJECT_ROOT/dist/$APP_NAME.app"
RELEASE_DIR="$PROJECT_ROOT/release"
RELEASE_APP="$RELEASE_DIR/$APP_NAME.app"
RELEASE_ZIP="$RELEASE_DIR/$APP_NAME-macOS-arm64.zip"
VERIFY_DIR="$PROJECT_ROOT/build/verification"

cd "$PROJECT_ROOT"

if [[ "$(uname -m)" != "arm64" ]]; then
  print -u2 "错误：此构建脚本只支持 Apple Silicon arm64 Mac。"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "错误：未找到 $PYTHON_BIN。请先创建 Python 3.12 虚拟环境。"
  exit 1
fi

"$PYTHON_BIN" -c \
  'import PyInstaller, gradio, torch, torchvision, segmentation_models_pytorch, timm'
shasum -a 256 -c MODEL_APP_CHECKSUM.sha256

rm -rf "$PROJECT_ROOT/build/ResidueCoverage"
rm -rf "$PROJECT_ROOT/dist/ResidueCoverage"
rm -rf "$DIST_APP"

"$PYTHON_BIN" -m PyInstaller \
  --clean \
  --noconfirm \
  "$PROJECT_ROOT/ResidueCoverage.spec"

if [[ ! -d "$DIST_APP" ]]; then
  print -u2 "错误：PyInstaller 未生成 $DIST_APP。"
  exit 1
fi

mkdir -p "$VERIFY_DIR"
codesign --verify --deep --strict --verbose=2 "$DIST_APP" \
  >"$VERIFY_DIR/codesign-verify.txt" 2>&1
codesign -dv --verbose=4 "$DIST_APP" \
  >"$VERIFY_DIR/codesign-details.txt" 2>&1

set +e
spctl --assess --type execute --verbose=4 "$DIST_APP" \
  >"$VERIFY_DIR/spctl.txt" 2>&1
SPCTL_STATUS=$?
set -e
print "$SPCTL_STATUS" >"$VERIFY_DIR/spctl-exit-code.txt"

"$PROJECT_ROOT/scripts/verify_macos_app.sh" "$DIST_APP" \
  >"$VERIFY_DIR/bundle-check.txt" 2>&1

mkdir -p "$RELEASE_DIR"
rm -rf "$RELEASE_APP"
rm -f "$RELEASE_ZIP"
rm -f "$RELEASE_DIR/SHA256SUMS.txt"
rm -f "$RELEASE_DIR/使用说明.txt"

ditto "$DIST_APP" "$RELEASE_APP"
ditto "$PROJECT_ROOT/使用说明.txt" "$RELEASE_DIR/使用说明.txt"
codesign --verify --deep --strict "$RELEASE_APP"

ditto -c -k --sequesterRsrc --keepParent "$RELEASE_APP" "$RELEASE_ZIP"

(
  cd "$RELEASE_DIR"
  shasum -a 256 \
    "$APP_NAME-macOS-arm64.zip" \
    "$APP_NAME.app/Contents/MacOS/$APP_NAME" \
    "$APP_NAME.app/Contents/Resources/models/$MODEL_NAME" \
    >SHA256SUMS.txt
)

print "构建完成：$RELEASE_APP"
print "交付 ZIP：$RELEASE_ZIP"
print "Gatekeeper 检查退出码：$SPCTL_STATUS（内部 ad-hoc 版本被拒绝属于预期）"
