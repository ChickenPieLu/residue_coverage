import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from pathlib import Path

import gradio as gr
import numpy as np

from predict import (
    DEFAULT_CHECKPOINT,
    choose_device,
    load_image,
    load_model,
    predict_mask,
)


device = choose_device("auto")
model = load_model(DEFAULT_CHECKPOINT, device)


def predict(image_path, threshold):
    image, image_tensor = load_image(Path(image_path))

    mask = predict_mask(
        model,
        image_tensor,
        device,
        threshold,
    )

    coverage = float(mask.mean() * 100)

    # Gradio显示图片时，转换成0～255的图像
    mask_image = mask.astype(np.uint8) * 255

    return mask_image, f"{coverage:.2f}%"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="上传一张图片"),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.05,
            label="预测阈值",
        ),
    ],
    outputs=[
        gr.Image(label="预测秸秆"),
        gr.Textbox(label="预测覆盖率"),
    ],
    title="秸秆覆盖率预测",
    description=(
        "上传一张（垂直于）农田的照片来预测"
    ),
)

demo.launch()