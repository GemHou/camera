#!/usr/bin/env python3
"""Local dev server with correct WASM MIME type."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".mjs": "text/javascript",
        ".js": "text/javascript",
    }

    def end_headers(self):
        path = self.path.split("?", 1)[0]
        if path.endswith((".html", ".js")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    import os

    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    print(f"Serving {ROOT} at http://127.0.0.1:8080/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
