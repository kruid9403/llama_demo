import json
import os
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import psycopg
from dotenv import load_dotenv

from hypothesis_monitor.engine import (
    build_result_context_blocks,
    review_prompt_for_similarity,
    summarize_references,
)


WEB_DIR = str(Path(__file__).resolve().parent / "web")
load_dotenv()

DB_URL = os.path.expandvars(
    os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
)
HYPOTHESIS_CONTEXT_MAX_CHARS = int(os.getenv("HYPOTHESIS_CONTEXT_MAX_CHARS", "24000"))
_GEN_ACTIVITY: dict[str, dict[str, Any]] = {}
_GEN_ACTIVITY_LOCK = threading.Lock()


def _activity_start(request_id: str, stage: str = "queued") -> None:
    with _GEN_ACTIVITY_LOCK:
        _GEN_ACTIVITY[request_id] = {
            "stage": stage,
            "started_at": time.time(),
            "updated_at": time.time(),
        }


def _activity_stage(request_id: str, stage: str) -> None:
    now = time.time()
    with _GEN_ACTIVITY_LOCK:
        row = _GEN_ACTIVITY.get(request_id)
        if not row:
            return
        row["stage"] = stage
        row["updated_at"] = now


def _activity_end(request_id: str) -> None:
    with _GEN_ACTIVITY_LOCK:
        _GEN_ACTIVITY.pop(request_id, None)


def _activity_payload(request_id: str) -> dict[str, Any]:
    now = time.time()
    with _GEN_ACTIVITY_LOCK:
        row = _GEN_ACTIVITY.get(request_id)
        active_count = len(_GEN_ACTIVITY)
    if not row:
        return {
            "request_id": request_id,
            "active": False,
            "active_count": active_count,
            "elapsed_seconds": 0.0,
            "stage": "completed_or_unknown",
        }
    started_at = float(row.get("started_at") or now)
    return {
        "request_id": request_id,
        "active": True,
        "active_count": active_count,
        "elapsed_seconds": max(0.0, now - started_at),
        "stage": row.get("stage") or "running",
    }


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
        parsed = urlparse(self.path)
        if parsed.path == "/api/hypothesis/stats":
            try:
                self._json_response(200, {"ok": True, "stats": _db_stats()})
            except Exception as exc:
                self._json_response(500, {"ok": False, "message": str(exc)})
            return
        if parsed.path == "/api/hypothesis/activity":
            request_id = (parse_qs(parsed.query).get("request_id") or [""])[0].strip()
            if not request_id:
                self._json_response(400, {"ok": False, "message": "Missing request_id"})
                return
            self._json_response(200, {"ok": True, "activity": _activity_payload(request_id)})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/hypothesis/generate":
            self.send_error(404, "Not found")
            return

        try:
            data = self._read_json_body()
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            self.send_error(400, "Missing prompt")
            return
        request_id = str(data.get("request_id", "")).strip() or uuid.uuid4().hex

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

        _activity_start(request_id, stage="review_prompt")
        try:
            review = review_prompt_for_similarity(prompt)
            search_query = str(review.get("search_query") or prompt)

            try:
                _activity_stage(request_id, "similarity_search")
                from vector_store.retrieval import similarity_search_with_references

                raw_results = similarity_search_with_references(
                    question=search_query,
                    document_urls=document_urls,
                    top_k=top_k,
                    recent_only=recent_only,
                )
                references = summarize_references(raw_results, limit=top_k)
                _activity_stage(request_id, "context_build")
                context_blocks = build_result_context_blocks(raw_results, max_chars=HYPOTHESIS_CONTEXT_MAX_CHARS)
                _activity_stage(request_id, "llama_generation")
                from hypothesis_monitor.llm import HypothesisNoveltyError, generate_hypothesis_from_context

                try:
                    hypothesis, llm_meta = generate_hypothesis_from_context(
                        user_prompt=prompt,
                        search_query=search_query,
                        context_blocks=context_blocks,
                    )
                    generation_mode = "llama3"
                except HypothesisNoveltyError as llm_exc:
                    self._json_response(
                        422,
                        {
                            "ok": False,
                            "message": str(llm_exc),
                            "generation_mode": "llama3_rejected_novelty",
                            "request_id": request_id,
                        },
                    )
                    return
                except Exception as llm_exc:
                    self._json_response(
                        503,
                        {
                            "ok": False,
                            "message": f"Llama generation failed: {llm_exc}",
                            "generation_mode": "llama3_failed",
                            "request_id": request_id,
                        },
                    )
                    return
            except Exception as exc:
                self._json_response(500, {"ok": False, "message": str(exc), "request_id": request_id})
                return

            self._json_response(
                200,
                {
                    "ok": True,
                    "prompt": prompt,
                    "review": review,
                    "search_query": search_query,
                    "top_k": top_k,
                    "recent_only": recent_only,
                    "hypothesis": hypothesis,
                    "references": references,
                    "retrieved_count": len(references),
                    "generation_mode": generation_mode,
                    "request_id": request_id,
                    "llm": llm_meta,
                },
            )
        finally:
            _activity_end(request_id)


def run(host: str = "127.0.0.1", port: int = 8030) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Hypothesis UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
