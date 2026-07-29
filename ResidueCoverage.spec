# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)


project_root = Path(SPECPATH).resolve()
model_name = "smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"

datas = [
    (str(project_root / model_name), "models"),
    (str(project_root / "MODEL_APP_CHECKSUM.sha256"), "models"),
    (str(project_root / "resources" / "gradio" / "hash_seed.txt"), "gradio"),
]

# Gradio loads its browser frontend and component modules dynamically.
datas += collect_data_files(
    "gradio",
    include_py_files=True,
    includes=[
        "templates/**/*",
        "_simple_templates/**/*",
        "media_assets/**/*",
        "icons/**/*",
        "_vendor/licenses/**/*",
        "**/*.py",
        "**/*.pyi",
        "package.json",
        "_workflow_curated_snapshot.json",
    ],
)
datas += collect_data_files("gradio_client")
datas += collect_data_files("safehttpx")
datas += collect_data_files("groovy")

for distribution in (
    "gradio",
    "gradio_client",
    "huggingface_hub",
    "safehttpx",
    "semantic_version",
    "segmentation-models-pytorch",
    "timm",
    "torch",
    "torchvision",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("gradio")
    + collect_submodules("segmentation_models_pytorch")
    + [
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
    ]
)

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cv2",
        "imagecodecs",
        "joblib",
        "matplotlib",
        "scipy",
        "sklearn",
        "tifffile",
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
    ],
    module_collection_mode={
        "torch._numpy": "py",
    },
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ResidueCoverage",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ResidueCoverage",
)

app = BUNDLE(
    collection,
    name="ResidueCoverage.app",
    icon=None,
    bundle_identifier="local.residuecoverage.internal",
    version="0.1.0",
    info_plist={
        "CFBundleDisplayName": "ResidueCoverage",
        "CFBundleName": "ResidueCoverage",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSApplicationCategoryType": "public.app-category.utilities",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    },
)
