#!/usr/bin/env python3
"""Serve the Slurm monitor dashboard and JSON APIs."""

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE = BASE_DIR / "state" / "jobs.json"
DEFAULT_HISTORY = BASE_DIR / "state" / "snapshots.jsonl"
DEFAULT_WEB_ROOT = BASE_DIR / "dashboard"


def read_json_file(path: Path) -> bytes:
    if not path.exists():
        payload = {
            "schema_version": 1,
            "summary": {
                "collected_at": None,
                "cost_label": "cost units",
                "jobs": 0,
                "active_jobs": 0,
                "running_jobs": 0,
                "pending_jobs": 0,
                "active_gpus": 0,
                "active_memory": "N/A",
                "active_cost_units": 0,
                "total_cost_units": 0,
                "errors": [f"state file not found: {path}"],
            },
            "jobs": [],
        }
        return json.dumps(payload).encode("utf-8")
    return path.read_bytes()


def read_history(path: Path, limit: int) -> bytes:
    if not path.exists():
        return b"[]"
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return json.dumps(records).encode("utf-8")


def make_handler(state_path: Path, history_path: Path, web_root: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "SlurmMonitorDashboard/1.0"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                self.send_bytes(read_json_file(state_path), "application/json", no_cache=True)
                return
            if parsed.path == "/api/history":
                self.send_bytes(read_history(history_path, 400), "application/json", no_cache=True)
                return
            if parsed.path == "/api/health":
                payload = {
                    "ok": state_path.exists(),
                    "state": str(state_path),
                    "history": str(history_path),
                    "mtime": state_path.stat().st_mtime if state_path.exists() else None,
                }
                self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", no_cache=True)
                return

            path = parsed.path
            if path in {"", "/"}:
                path = "/index.html"
            candidate = (web_root / path.lstrip("/")).resolve()
            try:
                candidate.relative_to(web_root.resolve())
            except ValueError:
                self.send_error(404)
                return
            if not candidate.exists() or not candidate.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            self.send_bytes(candidate.read_bytes(), content_type)

        def log_message(self, fmt: str, *args) -> None:
            print("%s - %s" % (self.address_string(), fmt % args), file=sys.stderr)

        def send_bytes(self, body: bytes, content_type: str, no_cache: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if no_cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local Slurm monitor dashboard.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="State JSON written by monitor.py.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="Snapshot JSONL written by monitor.py.")
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT, help="Dashboard static file directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    handler = make_handler(args.state, args.history, args.web_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving Slurm dashboard at {url}", flush=True)
    print(f"State: {args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
