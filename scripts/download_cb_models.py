#!/usr/bin/env python3
"""Download crop + beauty ONNX models for web deployment.
These are exported from the step5 and step1 training results.

Usage: python scripts/download_cb_models.py
"""
import os, shutil, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')

# Source paths (adjust if you retrained)
SOURCES = {
    'crop_model.onnx': '/root/autodl-tmp/gem_data/interim/FOV169/20260703_211104/crop_model.onnx',
    'beauty_score.onnx': '/root/autodl-tmp/gem_data/interim/beauty_both_resnet18_nocoord/20260705_185032/beauty_score.onnx',
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, src in SOURCES.items():
        dst = os.path.join(MODELS_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            sz = os.path.getsize(dst) / 1024 / 1024
            print(f'  {name}: {sz:.0f} MB  ->  {dst}')
        else:
            print(f'  {name}: NOT FOUND at {src}', file=sys.stderr)
            return 1
    print('\nDone. Models ready for web deployment.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
