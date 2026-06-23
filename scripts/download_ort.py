import re
import urllib.request
from pathlib import Path

BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/"
OUT = Path(__file__).resolve().parents[1] / "vendor" / "onnxruntime-web"
OUT.mkdir(parents=True, exist_ok=True)

html = urllib.request.urlopen(BASE).read().decode()
files = sorted(set(re.findall(r"/npm/onnxruntime-web@1\.19\.2/dist/([^\"]+)", html)))
wasm_and_js = [f for f in files if f.endswith((".wasm", ".mjs", ".js")) and "training" not in f]

for name in wasm_and_js:
    url = BASE + name
    path = OUT / name
    if path.exists() and path.stat().st_size > 0:
        print("skip", name)
        continue
    print("download", name)
    urllib.request.urlretrieve(url, path)
    print(" ", path.stat().st_size // 1024, "KB")

print("done:", OUT)
