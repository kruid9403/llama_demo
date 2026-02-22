import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv


WEB_DIR = str(Path(__file__).resolve().parent / "web")
load_dotenv()

DB_URL = os.path.expandvars(
    os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
)


def _db_stats() -> dict[str, Any]:
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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
        if self.path == "/api/similarity/stats":
            try:
                self._json_response(200, {"ok": True, "stats": _db_stats()})
            except Exception as exc:
                self._json_response(500, {"ok": False, "message": str(exc)})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/similarity/search":
            self.send_error(404, "Not found")
            return

        try:
            data = self._read_json_body()
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        query = str(data.get("query", "")).strip()
        if not query:
            self.send_error(400, "Missing query")
            return

        raw_urls = data.get("document_urls") or []
        if isinstance(raw_urls, list):
            document_urls = [str(u).strip() for u in raw_urls if str(u).strip()] or None
        else:
            document_urls = None

        try:
            top_k = int(data.get("top_k", 8))
        except (TypeError, ValueError):
            self.send_error(400, "top_k must be an integer")
            return
        top_k = max(1, min(top_k, 50))
        recent_only = _to_bool(data.get("recent_only", False))

        try:
            from vector_store.retrieval import similarity_search_with_references

            results = similarity_search_with_references(
                question=query,
                document_urls=document_urls,
                top_k=top_k,
                recent_only=recent_only,
            )
        except Exception as exc:
            self._json_response(500, {"ok": False, "message": str(exc)})
            return

        self._json_response(
            200,
            {
                "ok": True,
                "query": query,
                "top_k": top_k,
                "recent_only": recent_only,
                "results": results,
            },
        )


def run(host: str = "127.0.0.1", port: int = 8020) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Similarity UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
