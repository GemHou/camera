"""Run Depth Anything V2 Small ONNX inference on a local image."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


MODEL_REPO = "onnx-community/depth-anything-v2-small"
MODEL_FILE = "onnx/model.onnx"
INPUT_SIZE = 518


def load_model() -> ort.InferenceSession:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    providers = ort.get_available_providers()
    preferred = []
    if "CUDAExecutionProvider" in providers:
        preferred.append("CUDAExecutionProvider")
    preferred.append("CPUExecutionProvider")
    return ort.InferenceSession(model_path, providers=preferred)


def preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    return tensor, (h, w)


def postprocess(depth: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    depth = np.squeeze(depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = (depth * 255.0).astype(np.uint8)
    depth = cv2.resize(depth, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)


def run(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    session = load_model()
    input_name = session.get_inputs()[0].name

    tensor, original_size = preprocess(image)
    start = time.perf_counter()
    depth = session.run(None, {input_name: tensor})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    depth_color = postprocess(depth, original_size)
    combined = np.hstack([image, depth_color])
    cv2.imwrite(str(output_path), combined)

    depth_only_path = output_path.with_name(output_path.stem.replace("_depth", "") + "_depth_only" + output_path.suffix)
    cv2.imwrite(str(depth_only_path), depth_color)

    print(f"Input: {image_path}")
    print(f"Output: {output_path}")
    print(f"Depth only: {depth_only_path}")
    print(f"Inference: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, default=Path("20260622-094912.jpg"), nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=Path("20260622-094912_depth.jpg"))
    args = parser.parse_args()
    run(args.image, args.output)
