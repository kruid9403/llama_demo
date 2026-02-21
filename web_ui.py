import json
import os
import queue
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import psycopg

from research_debate import run_research_debate


WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")


class Handler(SimpleHTTPRequestHandler):
    active_stop_event: threading.Event | None = None
    active_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _json_response(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _vector_stats(self) -> dict:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks;")
                chunks = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM documents;")
                documents = cur.fetchone()[0]
                cur.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'));")
                chunk_table_size = cur.fetchone()[0]
                cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
                database_size = cur.fetchone()[0]
        return {
            "chunks": chunks,
            "documents": documents,
            "chunk_table_size": chunk_table_size,
            "database_size": database_size,
        }

    def do_GET(self):
        if self.path == "/api/stats":
            try:
                self._json_response(200, {"ok": True, "stats": self._vector_stats()})
            except Exception as exc:
                self._json_response(500, {"ok": False, "error": str(exc)})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/interrupt":
            with Handler.active_lock:
                if Handler.active_stop_event is not None:
                    Handler.active_stop_event.set()
                    self._json_response(200, {"ok": True, "message": "Interrupt requested."})
                    return
            self._json_response(200, {"ok": True, "message": "No active run."})
            return

        if self.path not in {"/api/ask", "/api/stream"}:
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        question = (data.get("question") or "").strip()
        history = data.get("history") or []
        url = (data.get("url") or "").strip() or None

        if not question:
            self.send_error(400, "Missing question")
            return

        with Handler.active_lock:
            if Handler.active_stop_event is not None and not Handler.active_stop_event.is_set():
                self.send_error(409, "Another debate is already running")
                return
            stop_event = threading.Event()
            Handler.active_stop_event = stop_event

        if self.path == "/api/ask":
            transcript = ""
            try:
                transcript = run_research_debate(
                    question=question,
                    history=history,
                    stream_queue=None,
                    stop_event=stop_event,
                    url=url,
                )
                self._json_response(200, {"answer": transcript})
            finally:
                with Handler.active_lock:
                    if Handler.active_stop_event is stop_event:
                        Handler.active_stop_event = None
            return

        token_queue: queue.Queue[str | None] = queue.Queue()

        def run_debate() -> None:
            try:
                run_research_debate(
                    question=question,
                    history=history,
                    stream_queue=token_queue,
                    stop_event=stop_event,
                    url=url,
                )
            except Exception as exc:
                token_queue.put(f"\n[[STATUS]] Debate failed: {exc}\n")
            finally:
                token_queue.put(None)
                with Handler.active_lock:
                    if Handler.active_stop_event is stop_event:
                        Handler.active_stop_event = None

        threading.Thread(target=run_debate, daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

        try:
            while True:
                token = token_queue.get()
                if token is None:
                    break
                self.wfile.write(token.encode("utf-8"))
                self.wfile.flush()
        except BrokenPipeError:
            stop_event.set()
            return


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Web UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
