import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg

from scholarly_spider import crawl_and_ingest


WEB_DIR = str(Path(__file__).resolve().parent / "web")
DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")


class SpiderManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.status = "Idle"
        self.config: dict[str, Any] = {}
        self.pages_ingested = 0
        self.chunks_inserted = 0
        self.errors = 0
        self.skips = 0
        self.logs: deque[dict[str, Any]] = deque(maxlen=1200)

    def _reset(self, config: dict[str, Any]) -> None:
        self.config = {
            **config,
            "raw_dir": str(config["raw_dir"]) if config.get("raw_dir") is not None else None,
        }
        self.pages_ingested = 0
        self.chunks_inserted = 0
        self.errors = 0
        self.skips = 0
        self.logs.clear()
        self.started_at = time.time()
        self.finished_at = None
        self.status = "Starting..."

    def _append_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "log"))
        message = str(event.get("message", ""))
        now = datetime.now(timezone.utc).isoformat()
        record = {"ts": now, "type": event_type, "message": message}
        record.update({k: v for k, v in event.items() if k not in {"type", "message"}})

        with self.lock:
            self.logs.append(record)

            if event_type == "start":
                self.status = "Running"
            elif event_type == "ingested":
                self.pages_ingested = int(event.get("pages_total", self.pages_ingested))
                self.chunks_inserted = int(event.get("chunks_total", self.chunks_inserted))
            elif event_type in {"fetch_error", "error"}:
                self.errors += 1
            elif event_type.endswith("_skip") or event_type.startswith("skip_"):
                self.skips += 1
            elif event_type == "interrupted":
                self.status = "Interrupted"
            elif event_type == "done":
                self.pages_ingested = int(event.get("pages_total", self.pages_ingested))
                self.chunks_inserted = int(event.get("chunks_total", self.chunks_inserted))
                interrupted = bool(event.get("interrupted", False))
                self.status = "Interrupted" if interrupted else "Completed"

    def start(self, config: dict[str, Any]) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "Spider is already running."
            self.running = True
            self.stop_event = threading.Event()
            self._reset(config)

        def _run() -> None:
            try:
                crawl_and_ingest(
                    seed_urls=config["seed_urls"],
                    allowed_domains=config["allowed_domains"],
                    max_pages=config["max_pages"],
                    max_depth=config["max_depth"],
                    delay_seconds=config["delay_seconds"],
                    respect_robots=config["respect_robots"],
                    raw_dir=config["raw_dir"],
                    event_callback=self._append_event,
                    stop_event=self.stop_event,
                )
            except Exception as exc:
                self._append_event({"type": "error", "message": f"[error] {exc}", "error": str(exc)})
                with self.lock:
                    self.status = "Error"
            finally:
                with self.lock:
                    self.running = False
                    self.finished_at = time.time()
                    self.worker = None
                    self.stop_event = None

        worker = threading.Thread(target=_run, daemon=True)
        with self.lock:
            self.worker = worker
        worker.start()
        return True, "Spider started."

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            if not self.running or self.stop_event is None:
                return False, "No active spider run."
            self.stop_event.set()
        return True, "Stop requested."

    def _db_stats(self) -> dict[str, Any]:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                documents = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chunks;")
                chunks = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM document_references;")
                refs = cur.fetchone()[0]
                cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
                database_size = cur.fetchone()[0]
                cur.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'));")
                chunks_total_size = cur.fetchone()[0]
        return {
            "documents": documents,
            "chunks": chunks,
            "references": refs,
            "database_size": database_size,
            "chunks_total_size": chunks_total_size,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            started_at = self.started_at
            finished_at = self.finished_at
            runtime_seconds = 0.0
            if started_at is not None:
                end = time.time() if self.running else (finished_at or time.time())
                runtime_seconds = max(0.0, end - started_at)

            payload = {
                "running": self.running,
                "status": self.status,
                "pages_ingested": self.pages_ingested,
                "chunks_inserted": self.chunks_inserted,
                "errors": self.errors,
                "skips": self.skips,
                "started_at": started_at,
                "finished_at": finished_at,
                "runtime_seconds": round(runtime_seconds, 1),
                "config": self.config,
                "logs": list(self.logs)[-400:],
            }

        try:
            payload["db_stats"] = self._db_stats()
        except Exception as exc:
            payload["db_error"] = str(exc)
        return payload


MANAGER = SpiderManager()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _json_response(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        if self.path == "/api/spider/status":
            self._json_response(200, {"ok": True, "state": MANAGER.snapshot()})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/spider/stop":
            ok, message = MANAGER.stop()
            self._json_response(200 if ok else 409, {"ok": ok, "message": message})
            return

        if self.path != "/api/spider/start":
            self.send_error(404, "Not found")
            return

        try:
            data = self._read_json_body()
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        seed_urls = [str(u).strip() for u in (data.get("seed_urls") or []) if str(u).strip()]
        if not seed_urls:
            self.send_error(400, "seed_urls must include at least one URL")
            return

        domains = [str(d).strip().lower() for d in (data.get("allowed_domains") or []) if str(d).strip()]
        if not domains:
            domains = sorted(
                {(urlparse(u).hostname or "").lower() for u in seed_urls if (urlparse(u).hostname or "").strip()}
            )

        raw_dir_value = data.get("raw_dir")
        raw_dir_text = raw_dir_value.strip() if isinstance(raw_dir_value, str) else ""

        config = {
            "seed_urls": seed_urls,
            "allowed_domains": domains,
            "max_pages": int(data.get("max_pages", 0)),
            "max_depth": int(data.get("max_depth", -1)),
            "delay_seconds": float(data.get("delay_seconds", 1.0)),
            "respect_robots": bool(data.get("respect_robots", True)),
            "raw_dir": Path(raw_dir_text).expanduser() if raw_dir_text else None,
        }

        ok, message = MANAGER.start(config)
        self._json_response(200 if ok else 409, {"ok": ok, "message": message})


def run(host: str = "127.0.0.1", port: int = 8010) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Spider UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
