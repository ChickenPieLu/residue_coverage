"""Local-only Gradio application for ResidueCoverage."""

from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser

# Route PyInstaller's resource-tracker/worker invocations before heavy imports.
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Configure offline/local behavior before importing Gradio or model libraries.
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr
import numpy as np

from inference import (
    InferenceError,
    ResiduePredictor,
    get_default_predictor,
    model_load_count,
)


APP_NAME = "ResidueCoverage"
SERVER_NAME = "127.0.0.1"
PORT_START = 7860
PORT_END = 7959
EXIT_DELAY_SECONDS = 2.0

_LOGGER = logging.getLogger(APP_NAME)
_INSTANCE_LOCK: "InstanceLock | None" = None


def _runtime_directory() -> Path:
    override = os.environ.get("RESIDUE_COVERAGE_RUNTIME_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / "Library" / "Application Support" / APP_NAME


def _configure_logging() -> None:
    if _LOGGER.handlers:
        return
    try:
        log_directory = _runtime_directory()
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / "ResidueCoverage.log"
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)


class InstanceLock:
    """Keep one server per macOS user and reopen its page on repeated launch."""

    def __init__(self, runtime_directory: Path) -> None:
        self.runtime_directory = runtime_directory
        self.lock_path = runtime_directory / "instance.lock"
        self.state_path = runtime_directory / "server.json"
        self._handle = None

    def acquire(self) -> bool:
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        self._handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._handle.close()
            self._handle = None
            return False
        return True

    def write_state(self, url: str) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"url": url, "pid": os.getpid(), "started_at": time.time()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def existing_url(self) -> str | None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        url = state.get("url")
        return url if isinstance(url, str) and url.startswith("http://127.0.0.1:") else None


class MacOSDefaultBrowser(webbrowser.BaseBrowser):
    """Open a URL through LaunchServices, honoring the macOS default browser."""

    def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
        del new, autoraise
        try:
            subprocess.run(
                ["/usr/bin/open", url],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            _LOGGER.exception("无法通过 macOS 默认浏览器打开 %s", url)
            return False
        _LOGGER.info("已请求 macOS 默认浏览器打开 %s", url)
        return True


def _register_macos_default_browser() -> None:
    if sys.platform == "darwin":
        webbrowser.register(
            "residue-coverage-macos",
            None,
            MacOSDefaultBrowser(),
            preferred=True,
        )


def _select_loopback_port() -> int:
    for port in range(PORT_START, PORT_END + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind((SERVER_NAME, port))
            except PermissionError as error:
                raise RuntimeError(
                    "无法打开本地端口，系统或安全策略拒绝了本地网络监听权限。"
                ) from error
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"无法启动本地界面：{PORT_START}–{PORT_END} 端口均被占用。"
    )


def _predict_for_ui(
    predictor: ResiduePredictor | None,
    startup_error: str | None,
    image_path: str | None,
    threshold: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[np.ndarray, str, str]:
    if startup_error:
        raise gr.Error(startup_error)
    if predictor is None:
        raise gr.Error("模型尚未加载，无法预测。")
    if not image_path:
        raise gr.Error("请先上传一张图片。")

    progress(0.1, desc="正在读取并预处理图片…")
    prediction_started = time.perf_counter()
    try:
        progress(0.35, desc="正在使用模型预测…")
        result = predictor.predict(image_path, float(threshold))
        progress(0.9, desc="正在计算覆盖率…")
    except InferenceError as error:
        _LOGGER.exception("预测失败")
        raise gr.Error(str(error)) from error
    except BaseException as error:
        _LOGGER.exception("未处理的预测错误")
        raise gr.Error(f"预测失败：{error}") from error

    _LOGGER.info(
        "预测完成 device=%s size=%dx%d coverage=%.10f seconds=%.3f",
        predictor.device,
        result.image_size[1],
        result.image_size[0],
        result.coverage,
        time.perf_counter() - prediction_started,
    )
    mask_image = result.mask.astype(np.uint8) * 255
    return (
        mask_image,
        f"{result.coverage:.2f}%",
        f"✅ 预测完成。mask 尺寸：{result.image_size[1]} × {result.image_size[0]}",
    )


def build_demo(
    predictor: ResiduePredictor | None = None,
    startup_error: str | None = None,
) -> gr.Blocks:
    device_text = str(predictor.device).upper() if predictor else "不可用"
    error_text = (
        f"\n\n⚠️ **启动错误：** {startup_error}" if startup_error else ""
    )

    image_input = gr.Image(
        type="filepath",
        sources=["upload"],
        label="上传图片（JPG / JPEG / PNG / BMP / TIFF / WebP）",
    )
    threshold_input = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.5,
        step=0.05,
        label="预测阈值",
    )
    mask_output = gr.Image(
        label="预测 mask（白色为秸秆）",
        image_mode="L",
        format="png",
        interactive=False,
        buttons=["fullscreen", "download"],
    )
    coverage_output = gr.Textbox(label="预测覆盖率", interactive=False)
    status_output = gr.Markdown("等待上传图片。")

    def run_prediction(
        image_path: str | None,
        threshold: float,
        progress: gr.Progress = gr.Progress(),
    ):
        return _predict_for_ui(
            predictor,
            startup_error,
            image_path,
            threshold,
            progress,
        )

    demo = gr.Interface(
        fn=run_prediction,
        inputs=[image_input, threshold_input],
        outputs=[mask_output, coverage_output, status_output],
        title="秸秆覆盖率预测（内部测试版）",
        description=(
            "上传一张垂直于农田拍摄的图片，点击“开始预测”查看二值 mask 和"
            f"秸秆覆盖率。当前运行设备：**{device_text}**。图片只在本机处理，"
            "不会上传到互联网。"
            f"{error_text}"
        ),
        submit_btn="开始预测",
        clear_btn="清空",
        flagging_mode="never",
        analytics_enabled=False,
        api_visibility="private",
        show_progress="full",
    )

    def request_exit() -> str:
        def shutdown() -> None:
            _LOGGER.info("收到退出请求，正在关闭本地服务器")
            try:
                demo.close(verbose=False)
            finally:
                os._exit(0)

        timer = threading.Timer(EXIT_DELAY_SECONDS, shutdown)
        timer.daemon = True
        timer.start()
        return "✅ 已收到退出请求。应用将在约 2 秒后关闭，此标签页随后可以关闭。"

    with demo:
        gr.Markdown(
            "完成使用后请点击下方按钮退出应用；仅关闭浏览器标签页不会停止本地服务。"
        )
        exit_status = gr.Markdown()
        exit_button = gr.Button("退出应用", variant="stop")
        exit_button.click(
            fn=request_exit,
            inputs=None,
            outputs=exit_status,
            queue=False,
            api_name=False,
            show_progress="hidden",
        )

    return demo


def _write_self_test_output(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _run_self_test(arguments: list[str]) -> int:
    if len(arguments) < 2:
        return 2
    output_path = Path(arguments[0]).expanduser().resolve()
    image_paths = [Path(value).expanduser().resolve() for value in arguments[1:]]
    started = time.perf_counter()
    try:
        predictor = get_default_predictor()
        loaded_at = time.perf_counter()
        predictions = []
        for image_path in image_paths:
            prediction_started = time.perf_counter()
            result = predictor.predict(image_path, 0.5)
            predictions.append(
                {
                    "image": str(image_path),
                    "mask_height": int(result.mask.shape[0]),
                    "mask_width": int(result.mask.shape[1]),
                    "coverage": result.coverage,
                    "prediction_seconds": time.perf_counter() - prediction_started,
                }
            )
        payload = {
            "ok": True,
            "device": str(predictor.device),
            "model_load_count": model_load_count(),
            "model_load_seconds": loaded_at - started,
            "total_seconds": time.perf_counter() - started,
            "predictions": predictions,
        }
        _write_self_test_output(output_path, payload)
        return 0
    except BaseException as error:
        message = str(error) if isinstance(error, InferenceError) else f"自检失败：{error}"
        _write_self_test_output(
            output_path,
            {
                "ok": False,
                "error": message,
                "model_load_count": model_load_count(),
                "total_seconds": time.perf_counter() - started,
            },
        )
        return 1


def main() -> None:
    global _INSTANCE_LOCK

    _configure_logging()
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        raise SystemExit(_run_self_test(sys.argv[2:]))

    _register_macos_default_browser()
    runtime_directory = _runtime_directory()
    _INSTANCE_LOCK = InstanceLock(runtime_directory)
    if not _INSTANCE_LOCK.acquire():
        existing_url = _INSTANCE_LOCK.existing_url()
        if existing_url:
            webbrowser.open(existing_url)
        return

    startup_started = time.perf_counter()
    predictor: ResiduePredictor | None = None
    startup_error: str | None = None
    try:
        predictor = get_default_predictor()
        _LOGGER.info(
            "模型加载完成 device=%s load_count=%d seconds=%.3f",
            predictor.device,
            model_load_count(),
            time.perf_counter() - startup_started,
        )
    except InferenceError as error:
        startup_error = str(error)
        _LOGGER.exception("模型加载失败")

    try:
        port = _select_loopback_port()
    except RuntimeError as error:
        _LOGGER.exception("本地端口选择失败")
        raise SystemExit(str(error)) from error

    url = f"http://{SERVER_NAME}:{port}"
    _INSTANCE_LOCK.write_state(url)
    demo = build_demo(predictor, startup_error)
    _LOGGER.info("启动本地界面 url=%s", url)
    demo.launch(
        inbrowser=os.environ.get("RESIDUE_COVERAGE_DISABLE_BROWSER") != "1",
        share=False,
        server_name=SERVER_NAME,
        server_port=port,
        show_error=True,
        quiet=True,
        enable_monitoring=False,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
