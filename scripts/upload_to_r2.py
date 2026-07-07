#!/usr/bin/env python3
"""Upload FP32 ONNX models to Cloudflare R2 for CF Pages deployment.
Usage: python scripts/upload_to_r2.py <r2_endpoint_url>
  r2_endpoint_url 例: https://pub-xxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.dev
"""
import os, sys, shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')

FILES = ['crop_model.onnx', 'beauty_score.onnx']


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    r2_base = sys.argv[1].rstrip('/')
    # Use curl for upload (requires R2 S3 credentials or public bucket)
    print(f'R2 endpoint: {r2_base}')
    print(f'Files to upload:')
    for f in FILES:
        path = os.path.join(MODELS_DIR, f)
        if not os.path.exists(path):
            print(f'  {f}: NOT FOUND at {path}')
            continue
        sz = os.path.getsize(path) / 1024 / 1024
        print(f'  {f}: {sz:.0f} MB -> {r2_base}/{f}')

    print()
    print('Upload with curl (requires R2 API credentials):')
    for f in FILES:
        path = os.path.join(MODELS_DIR, f)
        if os.path.exists(path):
            print(f'  curl -X PUT "{r2_base}/{f}" --data-binary "@{path}"')

    print()
    print('Or use rclone / AWS CLI with R2 S3-compatible credentials.')
    print('After upload, update MODEL_URL in index.html if needed.')


if __name__ == '__main__':
    sys.exit(main())
