#!/usr/bin/env python3
"""Local dev server with correct WASM MIME type."""

import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "0.0.0.0"
PORT = 8080


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
        if path.endswith(".onnx"):
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    import os

    os.chdir(ROOT)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    ip = local_ip()
    print(f"Serving {ROOT}")
    print(f"  本机:   http://127.0.0.1:{PORT}/")
    print(f"  手机:   http://{ip}:{PORT}/  （需同一 WiFi）")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
