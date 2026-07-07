"""Quantize crop + beauty ONNX models to INT8 for faster mobile download."""
import os, sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

TARGETS = {
    'crop_model.onnx': 'crop_model_int8.onnx',
    'beauty_score.onnx': 'beauty_score_int8.onnx',
}


def main():
    for src_name, dst_name in TARGETS.items():
        src = os.path.join(MODELS_DIR, src_name)
        dst = os.path.join(MODELS_DIR, dst_name)
        if not os.path.exists(src):
            print(f'Skip {src_name}: not found')
            continue

        src_sz = os.path.getsize(src) / 1024 / 1024
        print(f'Quantizing {src_name} ({src_sz:.0f} MB) → {dst_name}...')
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        dst_sz = os.path.getsize(dst) / 1024 / 1024
        ratio = dst_sz / src_sz * 100
        print(f'  Done: {dst_sz:.0f} MB ({ratio:.0f}% of original)')

    print('\nDone.')


if __name__ == '__main__':
    main()
