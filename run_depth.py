"""Run Depth Anything V2 Small ONNX inference on a local image."""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_CONFIG = SCRIPT_DIR / "models" / "runtime_config.json"
DEFAULT_MODEL = SCRIPT_DIR / "models" / "depth_anything_v2_small.onnx"
INPUT_SIZE = 518


def load_runtime_config() -> dict:
    if RUNTIME_CONFIG.is_file():
        return json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    return {}


def load_model(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python scripts/optimize_da2.py"
        )

    n_threads = threads if threads is not None else (os.cpu_count() or 4)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = n_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ort.get_available_providers()
    preferred = []
    if "CUDAExecutionProvider" in providers:
        preferred.append("CUDAExecutionProvider")
    preferred.append("CPUExecutionProvider")

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=preferred,
    )
    print(f"CPU threads (intra_op): {n_threads}")
    return session


def preprocess(image_bgr: np.ndarray, input_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
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


def run(
    image_path: Path,
    output_path: Path,
    model_path: Path,
    input_size: int,
    threads: int | None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    session = load_model(model_path, threads)
    input_name = session.get_inputs()[0].name

    tensor, original_size = preprocess(image, input_size)
    start = time.perf_counter()
    depth = session.run(None, {input_name: tensor})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    depth_color = postprocess(depth, original_size)
    combined = np.hstack([image, depth_color])
    cv2.imwrite(str(output_path), combined)

    depth_only_path = output_path.with_name(output_path.stem.replace("_depth", "") + "_depth_only" + output_path.suffix)
    cv2.imwrite(str(depth_only_path), depth_color)

    print(f"Model: {model_path.name} ({input_size}x{input_size})")
    print(f"Input: {image_path}")
    print(f"Output: {output_path}")
    print(f"Depth only: {depth_only_path}")
    print(f"Inference: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    cfg = load_runtime_config()
    default_threads = 1 if cfg.get("forceSingleThread") else None

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, default=Path("20260622-094912.jpg"), nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=Path("20260622-094912_depth.jpg"))
    parser.add_argument(
        "-m", "--model",
        type=Path,
        default=SCRIPT_DIR / cfg.get("modelUrl", DEFAULT_MODEL.name),
    )
    parser.add_argument(
        "-s", "--input-size",
        type=int,
        default=cfg.get("inputSize", INPUT_SIZE),
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=default_threads,
        help="CPU 推理线程数（runtime_config 默认 forceSingleThread=1）",
    )
    args = parser.parse_args()
    run(args.image, args.output, args.model, args.input_size, args.threads)
