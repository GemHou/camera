#!/usr/bin/env python3
"""Export + quantize DA2 ViT-S for faster single-core inference (no retraining)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
DA2_ROOT = ROOT / "vendor" / "Depth-Anything-V2"
MODELS = ROOT / "models"
WEIGHTS_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/"
    "resolve/main/depth_anything_v2_vits.pth"
)
WEIGHTS_PATH = MODELS / "depth_anything_v2_vits.pth"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


def download_weights() -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.is_file() and WEIGHTS_PATH.stat().st_size > 1_000_000:
        return WEIGHTS_PATH
    print(f"Downloading weights -> {WEIGHTS_PATH}")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


def load_torch_model() -> torch.nn.Module:
    sys.path.insert(0, str(DA2_ROOT))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, size: int, out_path: Path) -> Path:
    if size % 14 != 0:
        raise ValueError(f"input size must be multiple of 14, got {size}")

    dummy = torch.randn(1, 3, size, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Export ONNX {size}x{size} -> {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def quantize_dynamic(fp32_path: Path, int8_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"Dynamic quant -> {int8_path}")
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    return int8_path


def benchmark_onnx(model_path: Path, size: int, threads: int = 1, repeats: int = 5) -> float:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((1, 3, size, size), dtype=np.float32)

    # warmup
    session.run(None, {input_name: tensor})

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(None, {input_name: tensor})
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def write_runtime_config(model_path: Path, size: int, threads: int, bench_ms: float) -> None:
    rel = model_path.relative_to(ROOT).as_posix()
    config = {
        "modelUrl": rel,
        "inputSize": size,
        "cacheKey": f"{model_path.stem}_v1",
        "threads": threads,
        "benchmarkMsSingleCore": round(bench_ms, 1),
    }
    config_path = MODELS / "runtime_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[392, 252],
        help="Square input sizes (must be multiples of 14)",
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    download_weights()
    model = load_torch_model()

    best_path: Path | None = None
    best_size = 0
    best_ms = float("inf")

    print(f"\nBenchmark (Python ORT, {args.threads} thread(s), target <= 1000 ms):\n")
    for size in args.sizes:
        fp32 = MODELS / f"da2_vits_{size}_fp32.onnx"
        int8 = MODELS / f"da2_vits_{size}_int8.onnx"
        export_onnx(model, size, fp32)
        quantize_dynamic(fp32, int8)

        for path, label in [(fp32, "fp32"), (int8, "int8")]:
            ms = benchmark_onnx(path, size, threads=args.threads)
            mb = path.stat().st_size / 1024 / 1024
            ok = "OK" if ms <= 1000 else "--"
            print(f"  [{ok}] {label:4} {size}x{size}  {ms:7.1f} ms  {mb:5.1f} MB  {path.name}")
            if ms < best_ms:
                best_ms = ms
                best_path = path
                best_size = size

    if best_path is None:
        raise RuntimeError("No int8 model produced")

    write_runtime_config(best_path, best_size, args.threads, best_ms)
    print(f"\nSelected: {best_path.name} ({best_ms:.1f} ms @ {best_size}px, 1 core Python ORT)")
    print("Browser WASM is slower; refresh index.html after restart serve.py")


if __name__ == "__main__":
    main()
